import datetime
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import torch
from nnsight import NNsight
from tqdm import tqdm
from transformers import AutoTokenizer

from sae_lens.sae_bench.ravel.utils.generation_utils import generate_batched


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(0)


@dataclass
class Prompt:
    text: str
    template: str
    attribute: str
    entity: str
    context_split: str
    entity_split: str
    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    completion: Optional[str] = None
    is_correct: Optional[bool] = None


@dataclass
class AttributePrompt:
    attribute: str
    templates: List[str]


def timezone_name_to_utc_offset(name):
    try:
        offset = ZoneInfo(name).utcoffset(datetime.datetime.now()).seconds
        sign = "+"
        if offset // 3600 >= 12:
            offset = 24 * 3600 - offset
            sign = "-"
        fmt_offset = str(datetime.timedelta(seconds=offset)).rsplit(":", 1)[0]
        if fmt_offset.startswith("0") and offset >= 1800:
            fmt_offset = fmt_offset[1:]
        return f"{sign}{fmt_offset}"
    except Exception:
        return None


@dataclass
class RAVELEntityPromptData:
    prompts: Dict[str, Prompt] = field(default_factory=dict)
    entity_attributes: Dict[str, Dict[str, str]] = field(default_factory=dict)
    template_splits: Dict[str, str] = field(default_factory=dict)
    entity_splits: Dict[str, str] = field(default_factory=dict)
    attribute_prompts: List[AttributePrompt] = field(default_factory=list)

    @classmethod
    def from_files(cls, entity_type: str, data_dir: str, tokenizer):
        # Load data from files
        with open(
            os.path.join(
                data_dir, "base", f"ravel_{entity_type}_attribute_to_prompts.json"
            )
        ) as f:
            attribute_prompts_dict = json.load(f)
        with open(
            os.path.join(data_dir, "base", f"ravel_{entity_type}_prompt_to_split.json")
        ) as f:
            template_splits = json.load(f)
        with open(
            os.path.join(
                data_dir, "base", f"ravel_{entity_type}_entity_attributes.json"
            )
        ) as f:
            entity_attributes = json.load(f)
        with open(
            os.path.join(data_dir, "base", f"ravel_{entity_type}_entity_to_split.json")
        ) as f:
            entity_splits = json.load(f)

        # Create Prompt objects with tokenized inputs
        prompts = {}
        for x in tqdm(entity_attributes):
            for a, ts in attribute_prompts_dict.items():
                for t in ts:
                    text = t % x
                    encoded = tokenizer(
                        text,
                        return_tensors="pt",
                        padding="max_length",
                        max_length=32,
                        truncation=True,
                    )
                    prompts[text] = Prompt(
                        text=text,
                        template=t,
                        attribute=a,
                        entity=x,
                        context_split=template_splits[t],
                        entity_split=entity_splits[x],
                        input_ids=encoded["input_ids"].squeeze(),
                        attention_mask=encoded["attention_mask"].squeeze(),
                    )

        # Create AttributePrompt objects
        attribute_prompts = [
            AttributePrompt(attribute=k, templates=v)
            for k, v in attribute_prompts_dict.items()
        ]

        return cls(
            prompts=prompts,
            entity_attributes=entity_attributes,
            template_splits=template_splits,
            attribute_prompts=attribute_prompts,
        )

    def add_wikipedia_prompts(
        self, entity_type: str, data_dir: str, tokenizer, model: NNsight
    ):
        # Load Wikipedia prompts
        wiki_file_path = os.path.join(
            data_dir, "base", f"wikipedia_{entity_type}_entity_prompts.json"
        )
        with open(wiki_file_path, "r") as f:
            wiki_prompts = json.load(f)

        # Filter Wikipedia prompts to keep only those with exactly one '%s'
        filtered_wiki_prompts = {
            k: v for k, v in wiki_prompts.items() if k.count("%s") == 1
        }

        # Create Prompt objects for Wikipedia prompts
        wiki_prompt_objects = []
        for template, info in filtered_wiki_prompts.items():
            entity = info["entity"]
            if entity:
                text = template % entity
                encoded = tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=32,
                    truncation=True,
                )
                prompt = Prompt(
                    text=text,
                    template=template,
                    attribute="Other",
                    entity=entity,
                    context_split=info["split"],
                    entity_split="train",  # Assuming all Wikipedia entities are in train split
                    input_ids=encoded["input_ids"].squeeze(),
                    attention_mask=encoded["attention_mask"].squeeze(),
                )
                wiki_prompt_objects.append(prompt)
                self.prompts[text] = prompt
            else:
                for entity in self.get_entities(info["split"]):
                    text = template % entity
                    encoded = tokenizer(
                        text,
                        return_tensors="pt",
                        padding="max_length",
                        max_length=32,
                        truncation=True,
                    )
                    prompt = Prompt(
                        text=text,
                        template=template,
                        attribute="Other",
                        entity=entity,
                        context_split=info["split"],
                        entity_split=self.entity_splits[entity],
                        input_ids=encoded["input_ids"].squeeze(),
                        attention_mask=encoded["attention_mask"].squeeze(),
                    )
                    wiki_prompt_objects.append(prompt)
                    self.prompts[text] = prompt

        # Generate completions for Wikipedia prompts
        completions = generate_batched(
            model, tokenizer, wiki_prompt_objects, batch_size=64, max_new_tokens=8
        )

        # Add completions to Prompt objects
        for prompt, (_, completion) in zip(wiki_prompt_objects, completions):
            prompt.completion = completion[len(prompt.text) :]

        # Update template_splits with Wikipedia prompts
        for template, info in filtered_wiki_prompts.items():
            self.template_splits[template] = info["split"]

        # Add 'Other' to attribute_prompts if not already present
        if "Other" not in [ap.attribute for ap in self.attribute_prompts]:
            self.attribute_prompts.append(
                AttributePrompt(
                    attribute="Other", templates=list(filtered_wiki_prompts.keys())
                )
            )

        print(f"Added {len(filtered_wiki_prompts)} Wikipedia prompt templates")

    def get_prompts_by_split(self, context_split: str) -> List[Prompt]:
        return [
            prompt
            for prompt in self.prompts.values()
            if prompt.context_split == context_split
        ]

    def get_entities(self, split: Optional[str] = None) -> List[str]:
        """
        Get entities, optionally filtered by split.

        Args:
            split (Optional[str]): The split to filter entities by ('train', 'val', or 'test').
                                If None, return all entities.

        Returns:
            List[str]: A list of entity names.
        """
        if split is None:
            return list(self.entity_splits.keys())
        else:
            return [
                entity
                for entity, entity_split in self.entity_splits.items()
                if entity_split == split
            ]

    def get_prompt_by_text(self, text: str) -> Prompt:
        assert text in self.prompts.keys(), f'Prompt with text "{text}" not found'
        return self.prompts.get(text)

    def get_prompts_by_template(self, template: str) -> List[Prompt]:
        return [p for p in self.prompts.values() if p.template == template]

    def get_prompts_by_attribute(self, attribute: str) -> List[Prompt]:
        return [p for p in self.prompts.values() if p.attribute == attribute]

    def get_prompts_by_entity(self, entity: str) -> List[Prompt]:
        return [p for p in self.prompts.values() if p.entity == entity]

    def _filter_data(self, filtered_prompts: Dict[str, Prompt]):
        filtered_entities = set(prompt.entity for prompt in filtered_prompts.values())
        filtered_attributes = set(
            prompt.attribute for prompt in filtered_prompts.values()
        )
        filtered_templates = set(
            prompt.template for prompt in filtered_prompts.values()
        )

        filtered_entity_attributes = {
            entity: attrs
            for entity, attrs in self.entity_attributes.items()
            if entity in filtered_entities
        }

        filtered_attribute_prompts = [
            AttributePrompt(
                attribute=ap.attribute,
                templates=[t for t in ap.templates if t in filtered_templates],
            )
            for ap in self.attribute_prompts
            if ap.attribute in filtered_attributes
        ]

        filtered_template_splits = {
            t: context_split
            for t, context_split in self.template_splits.items()
            if t in filtered_templates
        }

        filtered_entity_splits = {
            entity: split
            for entity, split in self.entity_splits.items()
            if entity in filtered_entities
        }

        return RAVELEntityPromptData(
            prompts=filtered_prompts,
            entity_attributes=filtered_entity_attributes,
            template_splits=filtered_template_splits,
            entity_splits=filtered_entity_splits,
            attribute_prompts=filtered_attribute_prompts,
        )

    def downsample(self, n: int):
        sampled_keys = random.sample(list(self.prompts.keys()), n)
        sampled_prompts = {k: self.prompts[k] for k in sampled_keys}
        return self._filter_data(sampled_prompts)

    def evaluate_completion(self, prompt: Prompt, completion: str) -> bool:
        label = self.entity_attributes[prompt.entity][prompt.attribute]
        if not label:
            return False

        norm_label = label.lower()
        norm_out = completion.split('"')[0].strip(' "').replace("\\/", "/").lower()

        if len(norm_label) < len(norm_out):
            correct = norm_out.startswith(norm_label)
        else:
            correct = norm_label.startswith(norm_out)

        # Exceptions
        if re.search('coord|"lat"|"long"|latitude|coordinates|longitude', prompt.text):
            try:
                correct = (
                    abs(
                        float(norm_label.strip("-−"))
                        - float(re.findall(r"\d+", norm_out)[0])
                    )
                    <= 2
                )
            except:
                correct = False
        elif re.search("United States|United Kingdom", label):
            norm_label = label.strip().replace("the ", "")
            norm_out = completion.strip().replace("the ", "")
            correct = norm_out.startswith(norm_label) or norm_out.startswith("England")
        elif re.search("South Korea", label):
            correct = norm_out.startswith("korea") or norm_out.startswith("south korea")
        elif re.search("North America", label):
            correct = (
                norm_label in norm_out
                or norm_out == "na"
                or norm_out.startswith("america")
            )
        elif re.search("Mandarin", label):
            correct = norm_out in norm_label or norm_out == "chinese"
        elif re.search("language", prompt.text) and "," in norm_label:
            correct = any(lang in norm_out for lang in norm_label.split(","))
        elif re.search("UTC", prompt.text) and "/" in norm_label:
            norm_label = timezone_name_to_utc_offset(label)
            if norm_label:
                correct = norm_out.startswith(norm_label.split(":")[0])
                if not correct and re.search(r"[+\-]0\d", norm_out):
                    correct = norm_out.replace("0", "", 1).startswith(
                        norm_label.split(":")[0]
                    )
                # Summer daylight saving time
                if not correct and (
                    re.search(r"\-[5-8]", norm_label)
                    and label.startswith("America")
                    or re.search(r"\+[0-3]", norm_label)
                    and label.startswith("Europe")
                    or re.search(r"\+[0-3]", norm_label)
                    and label.startswith("Africa")
                ):
                    out_offset_match = re.search(r"[+\-]?(\d\d?):\d+", norm_out)
                    label_offset_match = re.search(r"[+\-]?(\d\d?):\d+", norm_label)
                    if out_offset_match and label_offset_match:
                        norm_out_offset = int(out_offset_match.group(1))
                        norm_label_offset = int(label_offset_match.group(1))
                        correct = (
                            norm_out_offset <= norm_label_offset + 1
                            and norm_out_offset >= norm_label_offset - 1
                        )
                if (
                    not correct
                    and re.search(r"[+\-](\d+)", norm_out)
                    and int(re.search(r"[+\-](\d+)", norm_out).group(1)) > 11
                ):
                    offset = 24 - int(re.search(r"[+\-](\d+)", norm_out).group(1))
                    correct = str(offset) in norm_label
            else:
                correct = False

        return correct

    def generate_completions(
        self,
        model: NNsight,
        tokenizer: AutoTokenizer,
        batch_size: int = 32,
        max_length: Optional[int] = None,
        prompt_max_length: int = 48,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ):

        all_prompts = list(self.prompts.values())
        completions = generate_batched(
            model,
            tokenizer,
            all_prompts,
            batch_size=batch_size,
            max_length=max_length,
            prompt_max_length=prompt_max_length,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        for prompt, (_, completion) in zip(all_prompts, completions):
            prompt.completion = completion[len(prompt.text) :]

    def evaluate_correctness(self):
        for prompt in self.prompts.values():
            if prompt.completion is not None:
                prompt.is_correct = self.evaluate_completion(prompt, prompt.completion)

    def filter_correct(self):
        correct_prompts = {
            text: prompt for text, prompt in self.prompts.items() if prompt.is_correct
        }
        return self._filter_data(correct_prompts)

    def get_accuracy_stats(self):
        entity_template_stats = {}
        for prompt in self.prompts.values():
            if prompt.is_correct is not None:
                key = (prompt.entity, prompt.template)
                if key not in entity_template_stats:
                    entity_template_stats[key] = {"correct": 0, "total": 0}
                entity_template_stats[key]["total"] += 1
                if prompt.is_correct:
                    entity_template_stats[key]["correct"] += 1

        return entity_template_stats

    def filter_prompts_by_template_format(self):
        return {
            text: prompt
            for text, prompt in self.prompts.items()
            if prompt.template.count("%s") == 1
        }

    def filter_top_entities_and_templates(
        self, top_n_entities=400, top_n_templates_per_attribute=12
    ):
        stats = self.get_accuracy_stats()

        # Calculate entity scores
        entity_scores = {}
        for (entity, _), stat in stats.items():
            if entity not in entity_scores:
                entity_scores[entity] = 0
            entity_scores[entity] += stat["correct"]

        # Keep top N entities
        kept_entities = set(
            sorted(entity_scores, key=entity_scores.get, reverse=True)[:top_n_entities]
        )

        # Calculate template scores and keep top N per attribute
        template_scores = {}
        for (_, template), stat in stats.items():
            if template not in template_scores:
                template_scores[template] = 0
            template_scores[template] += stat["correct"]

        kept_templates = set()
        for attr in set(prompt.attribute for prompt in self.prompts.values()):
            attr_templates = [t for t in self.attribute_prompts if t.attribute == attr][
                0
            ].templates
            kept_templates.update(
                sorted(
                    [t for t in attr_templates if t in template_scores],
                    key=template_scores.get,
                    reverse=True,
                )[:top_n_templates_per_attribute]
            )

        # Filter prompts
        filtered_prompts = {
            text: prompt
            for text, prompt in self.prompts.items()
            if prompt.entity in kept_entities and prompt.template in kept_templates
        }

        return self._filter_data(filtered_prompts)

    def calculate_average_accuracy(self):
        correct = sum(1 for prompt in self.prompts.values() if prompt.is_correct)
        total = len(self.prompts)
        return correct / total if total > 0 else 0

    def __len__(self) -> int:
        return len(self.prompts)
