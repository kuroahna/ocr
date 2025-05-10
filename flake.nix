{
  description = "ocr";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    crane.url = "github:ipetkov/crane";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      flake-utils,
      nixpkgs,
      crane,
      rust-overlay,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };

        targetTriple = "x86_64-unknown-linux-gnu";

        toolchain =
          p:
          p.rust-bin.stable.latest.default.override {
            targets = [ targetTriple ];
          };
        craneLib = (crane.mkLib pkgs).overrideToolchain toolchain;

        src = craneLib.cleanCargoSource ./.;

        commonArgs = {
          inherit src;
          strictDeps = true;
        };

        cargoArtifacts = craneLib.buildDepsOnly (
          commonArgs
          // {
            nativeBuildInputs = with pkgs; [
              openssl
              pkg-config
            ];
            buildPhaseCargoCommand = ''
              cargo check --profile release --frozen --target ${targetTriple}
              cargo build --profile release --frozen --target ${targetTriple} --workspace
            '';
            checkPhaseCargoCommand = ''
              cargo test --profile release --frozen --target ${targetTriple} --workspace --no-run
            '';
          }
        );

        ocr = craneLib.buildPackage (
          commonArgs
          // {
            inherit cargoArtifacts;
            cargoExtraArgs = "--frozen --target ${targetTriple} --workspace";
            nativeBuildInputs = with pkgs; [
              openssl
              pkg-config
            ];
          }
        );
      in
      {
        formatter = pkgs.nixfmt-rfc-style;
        packages.default = ocr;
        checks = {
          inherit ocr;

          clippy = craneLib.cargoClippy (
            commonArgs
            // {
              cargoArtifacts = cargoArtifacts;
              cargoClippyExtraArgs = "--target ${targetTriple} -- --deny warnings";
            }
          );

          format = craneLib.cargoFmt {
            src = src;
          };

          toml_format = craneLib.taploFmt {
            src = pkgs.lib.sources.sourceFilesBySuffices src [ ".toml" ];
          };

          deny = craneLib.cargoDeny { src = src; };
        };
        devShells.default = craneLib.devShell {
          checks = self.checks.${system};

          packages = with pkgs; [
            rust-analyzer
          ];

          # fixes: the cargo feature `public-dependency` requires a nightly
          # version of Cargo, but this is the `stable` channel
          #
          # This enables unstable features with the stable compiler
          # Remove once this is fixed in stable
          #
          # https://github.com/rust-lang/rust/issues/112391
          # https://github.com/rust-lang/rust-analyzer/issues/15046
          RUSTC_BOOTSTRAP = 1;
        };
      }
    );
}
