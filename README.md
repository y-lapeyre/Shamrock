<picture>
   <source media="(prefers-color-scheme: dark)" srcset="doc/shamrock-doc/src/images/no_background_nocolor.png"  width="600">
   <img alt="text" src="doc/logosham_white.png" width="600">
 </picture>

CI status:
![badge1](https://github.com/tdavidcl/Shamrock/actions/workflows/main.yml/badge.svg?branch=main)

DPC++ status :
![badge2](https://badgen.net/static/DPC++%2FCUDA/yes/green)
![badge2](https://badgen.net/static/DPC++%2FHIP:ROCM/yes/green)

AdaptiveCPP status :
![badge2](https://badgen.net/static/ACPP%2FOpenMP/yes/green)
![badge2](https://badgen.net/static/ACPP%2FROCM/yes/green)
![badge2](https://badgen.net/static/ACPP%2FCUDA/yes/green)

# Getting in touch

Join us on [Discord](https://discord.gg/Q69s5buyr5)! Alternatively, open a discussion or issue in this repository.

# Contributing

Shamrock accept contributions through github pull request :
1. Code contributions via [Pull request](https://github.com/tdavidcl/Shamrock/compare)
1. Documentation contributions via [Pull request](https://github.com/tdavidcl/Shamrock/compare)
3. Issue report via [Github issues](https://github.com/tdavidcl/Shamrock/issues/new/choose)

# Documentation

We provide both a book like documentation and the more classic doxygen style for more details about the sources
 - The documentation is available here : [mdbook doc](https://tdavidcl.github.io/Shamrock/mdbook/index.html)
 - The doxygen doc is available here : [doxygen](https://tdavidcl.github.io/Shamrock/doxygen/index.html)

# Getting started

```bash
git clone --recurse-submodules git@github.com:tdavidcl/Shamrock.git
```

## SYCL configuration

```bash
git clone --recurse-submodules git@github.com:Shamrock-code/ShamrockWorkspace.git
cd ShamrockWorkspace
source quickstart.sh
```


