
FROM mambaorg/micromamba:2.3.3-debian13-slim

RUN micromamba install bioconda::nanocomp && \
	micromamba clean -a -y

ENV PATH="/opt/conda/envs/nanocomp/bin:${PATH}"

# Minimal default command - Nextflow will normally override this when running inside the container
CMD ["bash"]

