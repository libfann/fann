# Fast Artificial Neural Network Library (Fork)
Please read the [README.md](https://github.com/libfann/fann) file from the original repo: [README.md](https://github.com/libfann/fann)

## Regarding this fork
I found some CMakeLists issues, and I might decide to rewrite some small parts of the code in the future. Since the main repo seems somewhat idle, I created this fork to be able to help out with issues on the repository. I'll later try to do a pull request with the main repo.

The most efficient way to contribute to this repository is by creating pull requests. Issues are welcome obviusly, but I'm not as skilled as I wish I was.

## Git-submodule + CMakeLists
#### Git-submodule
Add the repo to your desired folder, I will use the folder name `external/` in this example, like so:
`git submodule add https://github.com/sciencefyll/fann external/fann`

Then initiate it by using:
`git submodule update --init`

If you have made changes, stash them before you pull or you'll get some trouble:
`git submodule external/fann git stash`

And fetch the latest version:
`git submodule external/fann git pull origin master`

You could use a script to keep ur git-submodules up to date:
```bash
#!/bin/bash

git submodule update --init
git submodule foreach git stash
git submodule foreach git pull origin master
```

#### CMakeLists
Assuming your project structure is as follows, and that every git-submodule is in the folder named `external`:
Project_root_dir/
- /CMakeLists.txt
- /external/fann/*
- /external/CMakeLists.txt
- /src/CMakeLists.txt

Only files and parts of files that are relevant will be shown.

CMakeLists.txt:
```CMakeLists
# Third party libraries in lib
add_subdirectory(external) # Git-submodule folder

# Our source code folder, usually called src/ or the name of the project
add_subdirectory(src)
```

external/CMakeLists.txt:
```CMakeLists
add_subdirectory(fann) # Fann, for simplified neural network development.
```

src/CMakeLists.txt:
```CMakeLists
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../external/fann/include
```

This will allow you to include fann files like this: `#include "fann/fann.h"`.
