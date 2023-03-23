(set -e
    # windows
    case "$(uname)" in
    MINGW*)
        # cmd "/C ci.bat"
        exit
        ;;
    esac
    # linux and darwin
    cmake .
    make
    ./tests/fann_tests
)
