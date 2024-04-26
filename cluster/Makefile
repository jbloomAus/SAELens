all:
	docker build -t py312:latest -f Dockerfile_py312 ..
	singularity build py312.sif docker-daemon://py312:latest
	singularity build --fakeroot -F sae_lens.sif sae_lens.def

clean:
	rm -f py312.sif sae_lens.sif

.PHONY: all clean
