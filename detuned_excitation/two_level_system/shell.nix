with import <nixpkgs> {};
# shell to compile with f2py if it doesnt work with the system-f2py for whatever reason 
let
  pythonEnv = python39.withPackages (ps: [
    ps.numpy
  ]);
in mkShell {
  buildInputs = [
    pythonEnv
    gfortran
    clang
    libcxx
    swig
    zlib
  ];
}
