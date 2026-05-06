# Changelog

## [0.5.1](https://github.com/lv416e/brepax/compare/v0.5.0...v0.5.1) (2026-05-06)


### Features

* **brep:** add OCCT BRepGProp ground-truth helper ([#59](https://github.com/lv416e/brepax/issues/59)) ([045a800](https://github.com/lv416e/brepax/commit/045a80079b76f7f8b6fe5d904e383cb6c0e3ad82))
* **brep:** add unsigned distance from a 3D point to a padded closed polyline ([#63](https://github.com/lv416e/brepax/issues/63)) ([0c07c73](https://github.com/lv416e/brepax/commit/0c07c73c5e6f18225177fd05b348cffe69038cd9))
* **brep:** end-to-end trim-aware cone-face SDF, matched against OCCT ([#75](https://github.com/lv416e/brepax/issues/75)) ([22bf116](https://github.com/lv416e/brepax/commit/22bf1162f8d8f7b03f01a3f91b25f9cc45991980))
* **brep:** end-to-end trim-aware cylinder-face SDF, matched against OCCT ([#69](https://github.com/lv416e/brepax/issues/69)) ([c605fc7](https://github.com/lv416e/brepax/commit/c605fc7fe79c27193a8dbf4e92aea607019e6e46))
* **brep:** end-to-end trim-aware plane-face SDF, matched against OCCT ([#67](https://github.com/lv416e/brepax/issues/67)) ([2fc6e48](https://github.com/lv416e/brepax/commit/2fc6e487f743fe8f986fe94adb565e8c23bd5cb4))
* **brep:** end-to-end trim-aware sphere-face SDF, matched against OCCT ([#71](https://github.com/lv416e/brepax/issues/71)) ([cff4be9](https://github.com/lv416e/brepax/commit/cff4be97cd4cb1d53ce6076be11aca970720a82e))
* **brep:** extract cone trim-frame data for Marschner composition ([#72](https://github.com/lv416e/brepax/issues/72)) ([20de8a4](https://github.com/lv416e/brepax/commit/20de8a4532c95f04e4b50f3b24c13a785d521e53))
* **brep:** extract cylinder trim-frame data for Marschner composition ([#68](https://github.com/lv416e/brepax/issues/68)) ([4144077](https://github.com/lv416e/brepax/commit/4144077e236d990dd78d5b786f83150342c20ec5))
* **brep:** extract plane trim-frame data for Marschner composition ([#66](https://github.com/lv416e/brepax/issues/66)) ([5121b61](https://github.com/lv416e/brepax/commit/5121b61b154838bce70a1d801d014d85ab535dc3))
* **brep:** extract sphere trim-frame data for Marschner composition ([#70](https://github.com/lv416e/brepax/issues/70)) ([6fb7d97](https://github.com/lv416e/brepax/commit/6fb7d97afea4f50b0e4552b08149ec8fa31dc2f3))
* **brep:** trim-aware signed-blend SDF composition per ADR-0018 ([#65](https://github.com/lv416e/brepax/issues/65)) ([b0183fd](https://github.com/lv416e/brepax/commit/b0183fda5658768159700188c0df86b2fb0582d2))
* **brep:** TrimmedCSGStump wires per-primitive Marschner trim-aware SDF ([#76](https://github.com/lv416e/brepax/issues/76)) ([604fc0e](https://github.com/lv416e/brepax/commit/604fc0e93427bfb402a8afd0a5ea3801a953ddd6))
* **primitives:** add analytical foot-of-perpendicular on primitive surfaces ([#62](https://github.com/lv416e/brepax/issues/62)) ([119ffdf](https://github.com/lv416e/brepax/commit/119ffdf7ed198da69d23a49151a2f5efa95d88dd))


### Documentation

* **adr:** ADR-0018 trim-aware surface SDF composition ([#64](https://github.com/lv416e/brepax/issues/64)) ([9b43be0](https://github.com/lv416e/brepax/commit/9b43be0ef08adde523c4764979c580da4984b891))

## [0.5.0](https://github.com/lv416e/brepax/compare/v0.4.2...v0.5.0) (2026-04-20)


### Features

* **brep:** add enable_compilation_cache public API ([#53](https://github.com/lv416e/brepax/issues/53)) ([725fbf9](https://github.com/lv416e/brepax/commit/725fbf96f1e4a38a623db6af3293d6452aeeb574))


### Performance Improvements

* **brep:** batch face dispatch by signature, single vmap per group ([#52](https://github.com/lv416e/brepax/issues/52)) ([9d799f3](https://github.com/lv416e/brepax/commit/9d799f351d98dd112051577f50893468cc9c2ba0))
* **brep:** share JIT cache across faces by surface type and BSpline signature ([#51](https://github.com/lv416e/brepax/issues/51)) ([8931f24](https://github.com/lv416e/brepax/commit/8931f2426345896e27a678e4ed56caba7ec12c6c))


### Documentation

* **adr:** ADR-0017 triangulation cold/subsequent/warm performance regions ([#57](https://github.com/lv416e/brepax/issues/57)) ([d587dc9](https://github.com/lv416e/brepax/commit/d587dc9cecb01fa58b6a23e28e6962aaec308506))
* **brep:** surface enable_compilation_cache in README and example 11 ([#56](https://github.com/lv416e/brepax/issues/56)) ([88ebb0e](https://github.com/lv416e/brepax/commit/88ebb0ea64201e7140fcb40f17c974f01787a4aa))
* clean up stale explanation docs and update architecture index ([#49](https://github.com/lv416e/brepax/issues/49)) ([5484588](https://github.com/lv416e/brepax/commit/54845886ffbf33d6cd5194ae2d8614dbe9ccbab6))


### Miscellaneous Chores

* release 0.5.0 ([#58](https://github.com/lv416e/brepax/issues/58)) ([310899a](https://github.com/lv416e/brepax/commit/310899a623bfb210ccb26107246c6df75d16cad9))

## [0.4.2](https://github.com/lv416e/brepax/compare/v0.4.1...v0.4.2) (2026-04-19)


### Features

* **experimental:** add differentiable PDE solvers on SDF domains ([#47](https://github.com/lv416e/brepax/issues/47)) ([204987e](https://github.com/lv416e/brepax/commit/204987e69aeb330ff70c05f9ce5c3b4f72539c62))


### Documentation

* update release process for auto-publish via release-please ([#45](https://github.com/lv416e/brepax/issues/45)) ([904caa9](https://github.com/lv416e/brepax/commit/904caa9319d5e8906672c3d2b329d6e933350851))

## [0.4.1](https://github.com/lv416e/brepax/compare/v0.4.0...v0.4.1) (2026-04-19)


### Features

* **brep:** add mesh-based SDF for BSpline-heavy DFM metrics ([#43](https://github.com/lv416e/brepax/issues/43)) ([802a47f](https://github.com/lv416e/brepax/commit/802a47f72ca207e3bf197d7ce3352851743fb1f9))

## [0.4.0](https://github.com/lv416e/brepax/compare/v0.3.0...v0.4.0) (2026-04-19)


### Features

* **brep:** add divergence theorem volume with OCCT mesh hybrid triangulation ([#36](https://github.com/lv416e/brepax/issues/36)) ([38d1e95](https://github.com/lv416e/brepax/commit/38d1e9543473132ae63a0517b68f03b4efd2f150))
* **brep:** add divergence_volume API, remove GWN volume, add ADR-0016 ([#37](https://github.com/lv416e/brepax/issues/37)) ([e521a44](https://github.com/lv416e/brepax/commit/e521a44fd550986d335a016cd9a5820bd86e32d6))
* **brep:** add parametric mesh evaluation for design optimization ([#39](https://github.com/lv416e/brepax/issues/39)) ([b18a222](https://github.com/lv416e/brepax/commit/b18a22272a7a4fb4d9b1332726c7ec893bf925e8))
* **metrics:** add differentiable curvature field metric ([#30](https://github.com/lv416e/brepax/issues/30)) ([4fc649e](https://github.com/lv416e/brepax/commit/4fc649e19c87e7f669165e10ef1d65fbddb12b1a))
* **nurbs:** add 2D trim polygon extraction and differentiable trim indicator ([#34](https://github.com/lv416e/brepax/issues/34)) ([b8043a3](https://github.com/lv416e/brepax/commit/b8043a3925c30ce1af9a712f7e7f642b7d3c855d))
* **nurbs:** add parametric trim bounds for BSplineSurface ([#33](https://github.com/lv416e/brepax/issues/33)) ([7982cbb](https://github.com/lv416e/brepax/commit/7982cbbc804c4d711f5a046135f73ba00f1096a2))


### Bug Fixes

* **nurbs:** correct BSpline SDF sign and prevent Newton divergence ([#35](https://github.com/lv416e/brepax/issues/35)) ([18cf05b](https://github.com/lv416e/brepax/commit/18cf05b3994e0942efeb69e8b20e0e8ed65b7963))


### Performance Improvements

* **nurbs:** coarse initial guess and reduced Newton iterations ([#32](https://github.com/lv416e/brepax/issues/32)) ([36d27d2](https://github.com/lv416e/brepax/commit/36d27d28384247bfd9f3fa77a52adc07a4f5c515))


### Documentation

* update README and index for v0.4.0 divergence theorem features ([#40](https://github.com/lv416e/brepax/issues/40)) ([8ab7917](https://github.com/lv416e/brepax/commit/8ab791749dc4e967437aeac1199262c50e53143b))


### Miscellaneous Chores

* release 0.4.0 ([#41](https://github.com/lv416e/brepax/issues/41)) ([136cba9](https://github.com/lv416e/brepax/commit/136cba964c8c0393e4c3e25cdb47f6823e73ff7a))

## [0.3.0](https://github.com/lv416e/brepax/compare/v0.2.0...v0.3.0) (2026-04-17)


### Features

* **brep:** add CSG-Stump DNF representation and evaluation ([#17](https://github.com/lv416e/brepax/issues/17)) ([d5a2f49](https://github.com/lv416e/brepax/commit/d5a2f496c4625f024c934c8680b7cbb697959b91))
* **brep:** add PMC-based CSG-Stump reconstruction ([#19](https://github.com/lv416e/brepax/issues/19)) ([3f278d3](https://github.com/lv416e/brepax/commit/3f278d38e2a264bc0dc532be24f43a74457e4a31))
* **brep:** enable CSG reconstruction with NURBS surfaces ([#27](https://github.com/lv416e/brepax/issues/27)) ([a53d688](https://github.com/lv416e/brepax/commit/a53d688711b782d0d533e3d0f8f381bbe6026daa))
* **brep:** PMC fixture validation, compaction, grouping, and analytical evaluation ([#20](https://github.com/lv416e/brepax/issues/20)) ([a03d5c3](https://github.com/lv416e/brepax/commit/a03d5c3f6dd6677e070eadd7a58bca690c7413ae))
* **metrics:** add differentiable draft angle violation metric ([#25](https://github.com/lv416e/brepax/issues/25)) ([e5e1f50](https://github.com/lv416e/brepax/commit/e5e1f50178b32f6ff79c402ae2d6dbe6c575c798))
* **metrics:** add differentiable surface area via SDF boundary integral ([#22](https://github.com/lv416e/brepax/issues/22)) ([5755b22](https://github.com/lv416e/brepax/commit/5755b22b56dfc1b2ffb9c97145a7d3f5b6243264))
* **metrics:** add differentiable wall thickness metrics ([#24](https://github.com/lv416e/brepax/issues/24)) ([2ae247a](https://github.com/lv416e/brepax/commit/2ae247ad7a70e90d43653d589c882edb22b78f58))
* **nurbs:** add differentiable B-spline SDF proof of concept ([#26](https://github.com/lv416e/brepax/issues/26)) ([3c62787](https://github.com/lv416e/brepax/commit/3c627879e170b364b9c644ad1f81e47159e1b42a))
* **nurbs:** add rational B-spline weights and slow test markers ([#28](https://github.com/lv416e/brepax/issues/28)) ([f7b95c3](https://github.com/lv416e/brepax/commit/f7b95c3af2d9294f2d9e9f3f04c41ac8011f02ce))


### Documentation

* add differentiable NURBS B-Rep feasibility assessment ([#23](https://github.com/lv416e/brepax/issues/23)) ([aaa8e00](https://github.com/lv416e/brepax/commit/aaa8e00b62190e52476fbb71c3456772a202377c))


### Miscellaneous Chores

* bump release target to v0.3.0 ([#21](https://github.com/lv416e/brepax/issues/21)) ([f03fef4](https://github.com/lv416e/brepax/commit/f03fef4f7f719387ff1d8ffbf4aaf3e9626f72e7))

## [0.2.0](https://github.com/lv416e/brepax/compare/v0.1.1...v0.2.0) (2026-04-17)


### Features

* **brep:** add CSG tree reconstruction for stock-minus-features patterns ([#11](https://github.com/lv416e/brepax/issues/11)) ([22326c5](https://github.com/lv416e/brepax/commit/22326c51127a8a815f3903cecabdacd711eac6b0))
* **brep:** add differentiable CSG tree evaluation ([#12](https://github.com/lv416e/brepax/issues/12)) ([d83b9e2](https://github.com/lv416e/brepax/commit/d83b9e278560a43395af9d7985df1d566d2dcaa8))
* **brep:** add face adjacency graph from B-Rep topology ([#9](https://github.com/lv416e/brepax/issues/9)) ([0535a6a](https://github.com/lv416e/brepax/commit/0535a6a6028920d7eac4d2ac9d7498cb20e66d55))
* **brep:** add face-to-primitive conversion from STEP surfaces ([06e3b5a](https://github.com/lv416e/brepax/commit/06e3b5a283f13f3c68ad64a0e25d564a0c1b5665))


### Bug Fixes

* **ci:** change publish trigger from release event to tag push ([cd60aaf](https://github.com/lv416e/brepax/commit/cd60aaf84608a5bca292241f63b8342d7bbc598c))


### Documentation

* add commit and merge conventions to CLAUDE.md ([#8](https://github.com/lv416e/brepax/issues/8)) ([d647dd3](https://github.com/lv416e/brepax/commit/d647dd3ef73ebcd550dda12a1f011b1e8d0b8e8e))
* add CSG reconstruction feasibility assessment ([#10](https://github.com/lv416e/brepax/issues/10)) ([b9851be](https://github.com/lv416e/brepax/commit/b9851be80cd7f915e2c8dfe830122205d183a3b8))
* add STEP-to-optimization end-to-end demo ([#13](https://github.com/lv416e/brepax/issues/13)) ([a921567](https://github.com/lv416e/brepax/commit/a921567ab1973bf728ffba254b68cdc92ab9150e))


### Miscellaneous Chores

* bump release target to v0.2.0 ([#15](https://github.com/lv416e/brepax/issues/15)) ([bd7a1cb](https://github.com/lv416e/brepax/commit/bd7a1cb169c69271dd8015d80aeb2d160eaf6b20))

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
