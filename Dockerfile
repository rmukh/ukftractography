FROM redhat/ubi9:9.5-1738643550

LABEL org.opencontainers.image.authors="Tashrif Billah <tbillah@bwh.harvard.edu>"

# set up working directory, redefine home directory, remain root user
WORKDIR /home/pnlbwh
ENV HOME=/home/pnlbwh
ENV USER="root"
ENV LANG=en_US.UTF-8
ENV CMAKE=3.31.0

# install required libraries
RUN ln -sf /usr/share/zoneinfo/America/New_York /etc/localtime && \
    yum -y install wget file bzip2 vim git make unzip fftw gcc-c++ \
    openssl-devel libX11-devel && \
    yum clean all && rm -rf /var/cache/yum

# install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE}/cmake-${CMAKE}.tar.gz && \
    tar -xzf cmake-${CMAKE}.tar.gz && \
    cd cmake-${CMAKE} && mkdir build && cd build && \
    ../bootstrap --parallel=4 && make -j4

# build ukftractography
RUN git clone https://github.com/pnlbwh/ukftractography.git && \
    cd ukftractography && mkdir build && cd build && \
    /home/pnlbwh/cmake-${CMAKE}/build/bin/cmake .. && make -j4

ENTRYPOINT ["/home/pnlbwh/ukftractography/build/UKFTractography-build/UKFTractography/bin/UKFTractography"]

