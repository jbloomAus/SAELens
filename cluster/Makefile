singularity-base: Dockerfile_py312
	docker build -t py312:latest -f Dockerfile_py312 ..
	singularity build py312.sif docker-daemon://py312:latest

singularity: singularity-base	
	singularity build --fakeroot -F sae_lens.sif sae_lens.def

all: singularity

clean:
	rm -f py312.sif sae_lens.sif

.PHONY: all clean docker-base singularity-base singularity
