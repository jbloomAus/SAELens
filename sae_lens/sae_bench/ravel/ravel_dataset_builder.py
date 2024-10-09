"""
RAVEL Entity Prompt Data Module

This module provides functionality for handling and processing entity prompt data
for the RAVEL evaluation benchmark.
"""

import datetime
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import torch
from nnsight import NNsight
from tqdm import tqdm
from transformers import AutoTokenizer

from sae_lens.sae_bench.ravel.utils.generation_utils import generate_batched


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(0)


@dataclass
class Prompt:
    """Represents a single prompt with its associated data."""

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
    """Represents an attribute with its associated prompt templates."""

    attribute: str
    templates: List[str]


@dataclass
class RAVELEntityPromptData:
    """
    Main class for handling RAVEL entity prompt data.

    This class provides methods for loading, processing, and evaluating
    entity prompt data for the RAVEL project.
    """

    prompts: Dict[str, Prompt] = field(default_factory=dict)
    entity_attributes: Dict[str, Dict[str, str]] = field(default_factory=dict)
    template_splits: Dict[str, str] = field(default_factory=dict)
    entity_splits: Dict[str, str] = field(default_factory=dict)
    attribute_prompts: List[AttributePrompt] = field(default_factory=list)

    @classmethod
    def from_files(cls, entity_type: str, data_dir: str, tokenizer):
        """
        Load RAVEL entity prompt data from files.

        Args:
            entity_type (str): Type of entity (e.g., 'person', 'place').
            data_dir (str): Directory containing the data files.
            tokenizer: Tokenizer to use for encoding prompts.

        Returns:
            RAVELEntityPromptData: Initialized instance with loaded data.
        """
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
        self, entity_type: str, data_dir: str, tokenizer: AutoTokenizer, model: NNsight
    ):
        """
        Add Wikipedia prompts to the existing prompts.

        Args:
            entity_type (str): Type of entity (e.g., 'person', 'place').
            data_dir (str): Directory containing the Wikipedia prompt file.
            tokenizer: Tokenizer to use for encoding prompts.
            model (NNsight): Model to use for generating completions.
        """
        # Load and filter Wikipedia prompts
        wiki_file_path = os.path.join(
            data_dir, "base", f"wikipedia_{entity_type}_entity_prompts.json"
        )
        with open(wiki_file_path, "r") as f:
            wiki_prompts = json.load(f)
        filtered_wiki_prompts = {
            k: v for k, v in wiki_prompts.items() if k.count("%s") == 1
        }

        # Create Prompt objects for Wikipedia prompts
        wiki_prompt_objects = []
        for template, info in filtered_wiki_prompts.items():
            entity = info["entity"]
            entities = [entity] if entity else self.get_entities(info["split"])
            for entity in entities:
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
                    entity_split=self.entity_splits.get(entity, "train"),
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

        # Update template_splits and attribute_prompts
        self.template_splits.update(
            {
                template: info["split"]
                for template, info in filtered_wiki_prompts.items()
            }
        )
        if "Other" not in [ap.attribute for ap in self.attribute_prompts]:
            self.attribute_prompts.append(
                AttributePrompt(
                    attribute="Other", templates=list(filtered_wiki_prompts.keys())
                )
            )

        print(f"Added {len(filtered_wiki_prompts)} Wikipedia prompt templates")

    def get_prompts_by_split(self, context_split: str) -> List[Prompt]:
        """Get prompts for a specific context split."""
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
        """Get a specific prompt by its text."""
        assert text in self.prompts, f'Prompt with text "{text}" not found'
        return self.prompts[text]

    def get_prompts_by_template(self, template: str) -> List[Prompt]:
        """Get all prompts for a specific template."""
        return [p for p in self.prompts.values() if p.template == template]

    def get_prompts_by_attribute(self, attribute: str) -> List[Prompt]:
        """Get all prompts for a specific attribute."""
        return [p for p in self.prompts.values() if p.attribute == attribute]

    def get_prompts_by_entity(self, entity: str) -> List[Prompt]:
        """Get all prompts for a specific entity."""
        return [p for p in self.prompts.values() if p.entity == entity]

    def _filter_data(self, filtered_prompts: Dict[str, Prompt]):
        """
        Create a new RAVELEntityPromptData instance with filtered data.

        Args:
            filtered_prompts (Dict[str, Prompt]): Dictionary of prompts to keep.

        Returns:
            RAVELEntityPromptData: New instance with filtered data.
        """
        filtered_entities = set(prompt.entity for prompt in filtered_prompts.values())
        filtered_attributes = set(
            prompt.attribute for prompt in filtered_prompts.values()
        )
        filtered_templates = set(
            prompt.template for prompt in filtered_prompts.values()
        )

        return RAVELEntityPromptData(
            prompts=filtered_prompts,
            entity_attributes={
                entity: attrs
                for entity, attrs in self.entity_attributes.items()
                if entity in filtered_entities
            },
            template_splits={
                t: split
                for t, split in self.template_splits.items()
                if t in filtered_templates
            },
            entity_splits={
                entity: split
                for entity, split in self.entity_splits.items()
                if entity in filtered_entities
            },
            attribute_prompts=[
                AttributePrompt(
                    attribute=ap.attribute,
                    templates=[t for t in ap.templates if t in filtered_templates],
                )
                for ap in self.attribute_prompts
                if ap.attribute in filtered_attributes
            ],
        )

    def downsample(self, n: int):
        """
        Create a downsampled version of the dataset.

        Args:
            n (int): Number of prompts to keep in the downsampled dataset.

        Returns:
            RAVELEntityPromptData: New instance with downsampled data.
        """
        sampled_keys = random.sample(list(self.prompts.keys()), n)
        sampled_prompts = {k: self.prompts[k] for k in sampled_keys}
        return self._filter_data(sampled_prompts)

    def evaluate_completion(self, prompt: Prompt, completion: str) -> bool:
        """
        Evaluate if a completion is correct for a given prompt.

        Args:
            prompt (Prompt): The prompt to evaluate.
            completion (str): The generated completion.

        Returns:
            bool: True if the completion is correct, False otherwise.
        """
        label = self.entity_attributes[prompt.entity][prompt.attribute]
        if not label:
            return False

        norm_label = label.lower()
        norm_out = completion.split('"')[0].strip(' "').replace("\\/", "/").lower()

        correct = (
            norm_out.startswith(norm_label)
            if len(norm_label) < len(norm_out)
            else norm_label.startswith(norm_out)
        )

        # Handle special cases
        if (
            "coord" in prompt.text
            or "latitude" in prompt.text
            or "longitude" in prompt.text
        ):
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
        elif any(country in label for country in ["United States", "United Kingdom"]):
            norm_label = label.strip().replace("the ", "")
            norm_out = completion.strip().replace("the ", "")
            correct = norm_out.startswith(norm_label) or norm_out.startswith("England")
        elif "South Korea" in label:
            correct = norm_out.startswith("korea") or norm_out.startswith("south korea")
        elif "North America" in label:
            correct = (
                norm_label in norm_out
                or norm_out == "na"
                or norm_out.startswith("america")
            )
        elif "Mandarin" in label:
            correct = norm_out in norm_label or norm_out == "chinese"
        elif "language" in prompt.text and "," in norm_label:
            correct = any(lang in norm_out for lang in norm_label.split(","))
        elif "UTC" in prompt.text and "/" in norm_label:
            correct = self._evaluate_utc_completion(label, norm_out)

        return correct

    def _evaluate_utc_completion(self, label: str, norm_out: str) -> bool:
        """Helper method to evaluate UTC-related completions."""
        norm_label = timezone_name_to_utc_offset(label)
        if not norm_label:
            return False

        correct = norm_out.startswith(norm_label.split(":")[0])
        if not correct and re.search(r"[+\-]0\d", norm_out):
            correct = norm_out.replace("0", "", 1).startswith(norm_label.split(":")[0])

        # Handle summer daylight saving time
        if not correct and self._is_summer_dst_case(norm_label, label):
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

        return correct

    def _is_summer_dst_case(self, norm_label: str, label: str) -> bool:
        """Check if the case is a summer daylight saving time scenario."""
        return (re.search(r"\-[5-8]", norm_label) and label.startswith("America")) or (
            re.search(r"\+[0-3]", norm_label)
            and (label.startswith("Europe") or label.startswith("Africa"))
        )

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
        """
        Generate completions for all prompts using the given model.

        Args:
            model (NNsight): The model to use for generation.
            tokenizer (AutoTokenizer): The tokenizer to use.
            batch_size (int): Batch size for generation.
            max_length (Optional[int]): Maximum length of the generated sequence.
            prompt_max_length (int): Maximum length of the prompt.
            max_new_tokens (Optional[int]): Maximum number of new tokens to generate.
            **kwargs: Additional keyword arguments for generation.
        """
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
        """Evaluate the correctness of all completions."""
        for prompt in self.prompts.values():
            if prompt.completion is not None:
                prompt.is_correct = self.evaluate_completion(prompt, prompt.completion)

    def filter_correct(self):
        """
        Create a new instance with only correct prompts.

        Returns:
            RAVELEntityPromptData: New instance with only correct prompts.
        """
        correct_prompts = {
            text: prompt for text, prompt in self.prompts.items() if prompt.is_correct
        }
        return self._filter_data(correct_prompts)

    def get_accuracy_stats(self):
        """
        Calculate accuracy statistics for each entity-template pair.

        Returns:
            Dict: A dictionary with entity-template pairs as keys and their stats as values.
        """
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
        """
        Filter prompts to keep only those with a single '%s' in the template.

        Returns:
            Dict[str, Prompt]: Filtered prompts.
        """
        return {
            text: prompt
            for text, prompt in self.prompts.items()
            if prompt.template.count("%s") == 1
        }

    def filter_top_entities_and_templates(
        self, top_n_entities=400, top_n_templates_per_attribute=12
    ):
        """
        Filter the dataset to keep only the top entities and templates.

        Args:
            top_n_entities (int): Number of top entities to keep.
            top_n_templates_per_attribute (int): Number of top templates to keep per attribute.

        Returns:
            RAVELEntityPromptData: New instance with filtered data.
        """
        stats = self.get_accuracy_stats()

        # Calculate entity scores and keep top N entities
        entity_scores = {}
        for (entity, _), stat in stats.items():
            entity_scores[entity] = entity_scores.get(entity, 0) + stat["correct"]
        kept_entities = set(
            sorted(entity_scores, key=entity_scores.get, reverse=True)[:top_n_entities]
        )

        # Calculate template scores and keep top N per attribute
        template_scores = {}
        for (_, template), stat in stats.items():
            template_scores[template] = (
                template_scores.get(template, 0) + stat["correct"]
            )

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
        """
        Calculate the average accuracy across all prompts.

        Returns:
            float: Average accuracy.
        """
        correct = sum(1 for prompt in self.prompts.values() if prompt.is_correct)
        total = len(self.prompts)
        return correct / total if total > 0 else 0

    def __len__(self) -> int:
        """Return the number of prompts in the dataset."""
        return len(self.prompts)


def timezone_name_to_utc_offset(name: str) -> Optional[str]:
    """
    Convert a timezone name to its UTC offset.

    Args:
        name (str): Timezone name.

    Returns:
        Optional[str]: UTC offset as a string, or None if conversion fails.
    """
    try:
        offset = ZoneInfo(name).utcoffset(datetime.datetime.now()).seconds
        sign = "+" if offset < 12 * 3600 else "-"
        if offset >= 12 * 3600:
            offset = 24 * 3600 - offset
        fmt_offset = str(datetime.timedelta(seconds=offset)).rsplit(":", 1)[0]
        if fmt_offset.startswith("0") and offset >= 1800:
            fmt_offset = fmt_offset[1:]
        return f"{sign}{fmt_offset}"
    except Exception:
        return None
