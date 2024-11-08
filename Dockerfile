FROM python:3.8-slim

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools



COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user unet_11-03-19-51.pt /opt/app/
RUN python -m piptools sync requirements.txt

VOLUME /input /output

COPY --chown=user:user processv1.py /opt/app/

ENTRYPOINT [ "python", "-m", "processv1" ]
