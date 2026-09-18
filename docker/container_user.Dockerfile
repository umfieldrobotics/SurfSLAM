ARG BASE_IMAGE=sethgi/surfslam:slam
FROM ${BASE_IMAGE}


ARG USER_NAME=frog
ARG USER_ID=1000


# Prevent anything requiring user input
ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=linux

ENV TZ=America
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN useradd -m -l -u ${USER_ID} -s /bin/bash ${USER_NAME} \
    && usermod -aG video ${USER_NAME} \
    && export PATH=$PATH:/home/${USER_NAME}/.local/bin

# Give them passwordless sudo
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers


# Switch to user to run user-space commands
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

RUN sudo chown -R ${USER_NAME} /home/${USER_NAME}

COPY ./entrypoint.sh /entrypoint.sh
RUN sudo chmod +x /entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]
