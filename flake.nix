{
  description = "GMM";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          # 👇 yolo
          config.allowUnfree = true;
        };
        # pkgs = nixpkgs.legacyPackages.${system};


        python = pkgs.python311;
        python_packages = python.withPackages(ps: with ps;[
          ipython
          matplotlib
          pandas
          pytorch-bin
          mypy
        ]);

      in {
        devShells.default =
          pkgs.mkShell {
            buildInputs = [ python python_packages ];
          };
      });
}
