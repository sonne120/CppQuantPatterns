![C++](https://img.shields.io/badge/C%2B%2B-00599C)
![Language](https://img.shields.io/github/languages/top/sonne120/CppQuantPatterns)
![Stars](https://img.shields.io/github/stars/sonne120/CppQuantPatterns)
![Forks](https://img.shields.io/github/forks/sonne120/CppQuantPatterns)
![Last commit](https://img.shields.io/github/last-commit/sonne120/CppQuantPatterns)
![Repo size](https://img.shields.io/github/repo-size/sonne120/CppQuantPatterns)

Monte Carlo option pricing engine in C++11 & C++20 — policy-based design vs concepts, virtual-free hot loop, parallel fork-join
```mermaid
flowchart TB
    Builder -->|creates| Mediator
    Mediator -->|paths| Pricer
    Mediator -->|status| MIS
    Source -->|market data| Mediator
    Sde -->|drift/diffusion| Fdm
    Fdm -->|advance step| Mediator
    Rng -->|N 0,1 variates| Fdm
```
