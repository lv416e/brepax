# Changelog

## [0.1.1](https://github.com/lv416e/brepax/compare/v0.1.0...v0.1.1) (2026-04-16)


### Features

* **io:** add STEP file reading, metadata extraction, and shape visualization ([448a4bc](https://github.com/lv416e/brepax/commit/448a4bc25cff85476c54720af9f3ef1171c5b8d0))


### Bug Fixes

* **ci:** add viz extra to test matrix and skip plot tests when matplotlib missing ([e2d23b5](https://github.com/lv416e/brepax/commit/e2d23b580ce8251e4eed8bbfb9aa4107a2531979))


### Documentation

* add reference pages, tutorials, and clean up skeleton placeholders ([bd2ddef](https://github.com/lv416e/brepax/commit/bd2ddefb8e9240c47ac62ad9784599ed9bfdfe11))

## 0.1.0 (2026-04-16)


### Features

* add mold direction optimizer demonstrator ([6384a16](https://github.com/lv416e/brepax/commit/6384a16095680ff0d2fa1ebdb80447a976275f0a))
* add Phase 1 roadmap, hybrid optimizer skeleton, update scope ([c51da75](https://github.com/lv416e/brepax/commit/c51da753e12db509b509b1e5b35bf05751336ade))
* add skeleton modules for boolean, stratification, topology, brep ([e58f1b5](https://github.com/lv416e/brepax/commit/e58f1b5d99e22b2020e69b344cd4b47e22a3c29a))
* **analytical:** add disk_disk closed-form ground truth ([aabb905](https://github.com/lv416e/brepax/commit/aabb90584347be1f277286ed5f98b1899bce2bce))
* **analytical:** add sphere_sphere closed-form ground truth ([e8d7306](https://github.com/lv416e/brepax/commit/e8d73060a714ccfecf93cb8b250eec5d0563dd8f))
* **benchmarks:** add Axis 1 gradient accuracy harness for Method (A) ([50dc31b](https://github.com/lv416e/brepax/commit/50dc31b3ed19826d8f9e77dfbf98532ce1f20dae))
* **benchmarks:** add Axis 3 optimization trajectory and ADR-0010 ([1dbb199](https://github.com/lv416e/brepax/commit/1dbb199995a28f24f13289d5bf1cd1f3b690eaa9))
* **benchmarks:** add Gate 3 vmap scaling benchmark ([7a68882](https://github.com/lv416e/brepax/commit/7a68882c4be700f75ed46fd2e023ce1599c75f1d))
* **boolean:** add subtract/intersect operations, Cylinder+Plane drilling demo ([8dfa9a8](https://github.com/lv416e/brepax/commit/8dfa9a8ee389401775584ee38d7444eac4a41418))
* **boolean:** generalize Method (C) to grid-based exact SDF Boolean ([ae86dd0](https://github.com/lv416e/brepax/commit/ae86dd0bc5da57c6f41e4b9f35611cef4c88402a))
* **boolean:** generalize stratum detection to arbitrary primitive pairs ([5d56e48](https://github.com/lv416e/brepax/commit/5d56e4858f0e81caf1c581b235f153716ae05204))
* **boolean:** implement Method (A) smooth-min union area ([d0fba3b](https://github.com/lv416e/brepax/commit/d0fba3b7b6319ce62bcc1fd0091492bb3a06d31d))
* **boolean:** implement Method (C) stratum-aware union area ([45bad27](https://github.com/lv416e/brepax/commit/45bad276a87a1e4c650112667e5943792e7ca587))
* **primitives:** add analytical volume() method, use in stratum dispatch ([277a9e1](https://github.com/lv416e/brepax/commit/277a9e182f2f7f348e06a4cae0bdbdbd48de0cff))
* **primitives:** add Cone, Torus, Box primitives with volume() ([f746fe2](https://github.com/lv416e/brepax/commit/f746fe2db7f49f996052d0e4705006a04fe42e3f))
* **primitives:** add Cylinder 3D primitive, fix CI slow test skip ([3bf670d](https://github.com/lv416e/brepax/commit/3bf670d6e4a54b122cc3cef0c3787d294995d634))
* **primitives:** add Disk primitive and Primitive ABC ([dc0f0c1](https://github.com/lv416e/brepax/commit/dc0f0c15d32886f06d7582515625049e7ad95737))
* **primitives:** add FiniteCylinder bounded primitive ([f10fd1f](https://github.com/lv416e/brepax/commit/f10fd1fad9c132a982bafa06edb9592f064a7321))
* **primitives:** add Plane 3D primitive ([38d85e7](https://github.com/lv416e/brepax/commit/38d85e79a1fc5de67caf2537283d570470513a16))
* **primitives:** add Sphere 3D primitive ([1a8d5bc](https://github.com/lv416e/brepax/commit/1a8d5bc6fbfed8ac2bb0aad2d3259bd427c1952f))


### Bug Fixes

* **boolean:** restore stratum label dispatch in generalized Method (C) ([f1503f9](https://github.com/lv416e/brepax/commit/f1503f9a2830d7a03f21e42d248f31ce9536bfd1))
* **ci:** add type annotations to custom_vjp callbacks for mypy --strict ([d5859bb](https://github.com/lv416e/brepax/commit/d5859bb9b4f417c8e00196eb0ea17071bb2a8020))
* **ci:** exclude _version.py from ruff format, fix mypy no-any-return ([e9d7173](https://github.com/lv416e/brepax/commit/e9d7173f4bc40d0e3eedb3550b2fc777056e6746))
* **ci:** exclude auto-generated _version.py from ruff, bump action versions ([96c56fb](https://github.com/lv416e/brepax/commit/96c56fb398d8898304fa5a1c9fb7e42942fe9592))
* **ci:** remove unused type-ignore, fix formatting, enforce pre-push checks ([a576b14](https://github.com/lv416e/brepax/commit/a576b1435ed08d15e03e7abf29038c041665be61))
* **ci:** use setup-uv python-version for test matrix ([8d58588](https://github.com/lv416e/brepax/commit/8d585888eac27c6603c20390393e902eb7c4e2cb))


### Documentation

* add 6 example notebooks with jupytext sources ([c542fd0](https://github.com/lv416e/brepax/commit/c542fd096516652987ce5784a6b4a6274fcf6ffc))
* add boundary proximity benchmark, gradient pitfall guide, gate criteria ([23cdcb8](https://github.com/lv416e/brepax/commit/23cdcb8bb4398cd80b13ba7ee8e3a71c39b4f34c))
* add Diataxis structure with ADRs 0001-0008 ([b212e4e](https://github.com/lv416e/brepax/commit/b212e4e7997c958bb43fc279348f34d0b7767a93))
* add gate evaluation report, ADR-0011, hybrid strategy explanation ([b3ce4d0](https://github.com/lv416e/brepax/commit/b3ce4d046f9bdcb384960f93c5f49266eeaf8d87))
* add Phase 1 Gate evaluation report ([14df0fa](https://github.com/lv416e/brepax/commit/14df0fa1a06bea6cebe8bc9462af536642a69ec0))
* add reference pages, tutorials, and clean up skeleton placeholders ([bd2ddef](https://github.com/lv416e/brepax/commit/bd2ddefb8e9240c47ac62ad9784599ed9bfdfe11))
* add TOI correction mathematical derivation ([222d8dc](https://github.com/lv416e/brepax/commit/222d8dc753d7830b0cf4a5d29527d30542357237))
* **adr:** accept ADR-0003 (scalar int label) and ADR-0004 (custom_vjp) ([d1fd6db](https://github.com/lv416e/brepax/commit/d1fd6db360da49e3d372f8c3beaaa26220d13840))
* **adr:** update ADR-0004 with boundary analysis, add ADR-0009 ([c0c29f1](https://github.com/lv416e/brepax/commit/c0c29f1408b0ac11b83857be1cd82e99c9d70eb0))

## Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).
