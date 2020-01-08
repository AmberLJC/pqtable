wget https://github.com/opencv/opencv/archive/4.2.0.zip
unzip 4.2.0.zip
cd opencv-4.2.0/
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DOPENCV_GENERATE_PKGCONFIG=ON -DCMAKE_INSTALL_PREFIX=/usr/local ..
make -j4
sudo make install

cd /etc/ld.so.conf.d
sudo sh -c 'echo "/usr/local/lib" > opencv4.conf'
sudo ldconfig

sudo cp -f /usr/local/lib/pkgconfig/opencv4.pc  /usr/lib/pkgconfig/




