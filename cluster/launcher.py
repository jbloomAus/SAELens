#!/usr/bin/env python3
from absl import app, flags
from lxm3 import xm, xm_cluster
from lxm3.contrib import ucl

_LAUNCH_ON_CLUSTER = flags.DEFINE_boolean(
    "launch_on_cluster", False, "Launch on cluster"
)
_GPU = flags.DEFINE_boolean("gpu", False, "If set, use GPU")
_RAM = flags.DEFINE_integer("ram", 16, "RAM in GB")
_HOURS = flags.DEFINE_integer("hours", 16, "Walltime in hours")
_SINGULARITY_CONTAINER = flags.DEFINE_string(
    "container", None, "Path to singularity container"
)
_ENTRYPOINT = flags.DEFINE_string(
    "entrypoint", "sae_lens.examples.train_sae", "Entrypoint"
)


def main(_):
    # use script name as experiment title
    script_name = _ENTRYPOINT.value.split(".")[-1]
    with xm_cluster.create_experiment(experiment_title=script_name) as experiment:

        # Define the job requirements
        ram = _RAM.value
        if _GPU.value:
            job_requirements = xm_cluster.JobRequirements(gpu=1, ram=ram * xm.GB)
        else:
            job_requirements = xm_cluster.JobRequirements(ram=ram * xm.GB)

        # Define the executor
        hours = _HOURS.value
        if _LAUNCH_ON_CLUSTER.value:
            # This is a special case for using SGE in UCL where we use generic
            # job requirements and translate to SGE specific requirements.
            # Non-UCL users, use `xm_cluster.GridEngine directly`.
            executor = ucl.UclGridEngine(
                job_requirements,
                walltime=hours * xm.Hr,
            )
        else:
            executor = xm_cluster.Local(job_requirements)

        # Define the python package
        entrypoint = _ENTRYPOINT.value
        spec = xm_cluster.PythonPackage(
            # NOTE: 'path' is a relative path to the launcher that contains
            # your python package (i.e. the directory that contains pyproject.toml)
            # Here, .../cluster --> .../
            path="..",
            # Entrypoint is the python module that you would like to
            # In the implementation, this is translated to
            #   python3 -m py_package.main
            entrypoint=xm_cluster.ModuleName(entrypoint),
        )

        # Wrap the python_package to be executing in a singularity container.
        singularity_container = _SINGULARITY_CONTAINER.value
        if singularity_container is not None:
            spec = xm_cluster.SingularityContainer(
                spec,
                image_path=singularity_container,
            )

        [executable] = experiment.package(
            [xm.Packageable(spec, executor_spec=executor.Spec())]
        )

        args = [
            {"sae_class_name": "SparseAutoencoder"},
            {"sae_class_name": "GatedSparseAutoencoder"},
        ]
        experiment.add(
            xm_cluster.ArrayJob(executable=executable, executor=executor, args=args)
        )


if __name__ == "__main__":
    app.run(main)
