# Fast Artificial Neural Network Library
## FANN

**Fast Artificial Neural Network (FANN) Library** is a free open source neural network library, which implements multilayer artificial neural networks in C with support for both fully connected and sparsely connected networks.

Cross-platform execution in both fixed and floating point are supported. It includes a framework for easy handling of training data sets. It is easy to use, versatile, well documented, and fast.

Bindings to more than 15 programming languages are available.

An easy to read introduction article and a reference manual accompanies the library with examples and recommendations on how to use the library.

Several graphical user interfaces are also available for the library.

## FANN Features

* Multilayer Artificial Neural Network Library in C
* Backpropagation training (RPROP, Quickprop, Batch, Incremental)
* Evolving topology training which dynamically builds and trains the ANN (Cascade2)
* Easy to use (create, train and run an ANN with just three function calls)
* Fast (up to 150 times faster execution than other libraries)
* Versatile (possible to adjust many parameters and features on-the-fly)
* Well documented (An easy to read introduction article, a thorough reference manual, and a 50+ page university report describing the implementation considerations etc.)
* Cross-platform (configure script for linux and unix, dll files for windows, project files for MSVC++ and Borland compilers are also reported to work)
* Several different activation functions implemented (including stepwise linear functions for that extra bit of speed)
* Easy to save and load entire ANNs
* Several easy to use examples
* Can use both floating point and fixed point numbers (actually both float, double and int are available)
* Cache optimized (for that extra bit of speed)
* Open source, but can still be used in commercial applications (licenced under LGPL)
* Framework for easy handling of training data sets
* Graphical Interfaces
* Language Bindings to a large number of different programming languages
* Widely used (approximately 100 downloads a day)

## To Install

### On Linux

#### From Source

First you'll want to clone the repository:

    git clone https://github.com/libfann/fann.git

Once that's finished, navigate to the Root directory. In this case it would be ./fann:

    cd ./fann

Then run CMake

    cmake .

After that, you'll need to use elevated privileges to install the library:

    sudo make install

That's it! If everything went right, you should see a lot of text, and FANN should be installed!

#### From Distribution Packages

Many Linux distributions carry precompiled FANN packages in their binary repositories.

See Repology for further information: [![Packaging status](https://repology.org/badge/tiny-repos/fann.svg)](https://repology.org/project/fann/versions).

On Debian derived distributions install the package [libfann2](https://packages.debian.org/search?searchon=names&keywords=libfann2) and potentially [libfann-dev](https://packages.debian.org/search?searchon=names&keywords=libfann-dev) if you desire to develop for the library. Fx. by issuing the following command in a root terminal:

    apt install libfann2 libfann-dev

### Building fann - Using vcpkg

You can download and install fann using the [vcpkg](https://github.com/Microsoft/vcpkg) dependency manager:

    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    ./bootstrap-vcpkg.sh
    ./vcpkg integrate install
    ./vcpkg install fann

The fann port in vcpkg is kept up to date by Microsoft team members and community contributors. If the version is out of date, please [create an issue or pull request](https://github.com/Microsoft/vcpkg) on the vcpkg repository.
