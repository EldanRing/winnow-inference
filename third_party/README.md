The build retrieves llama.cpp at the revision in `runtime.lock.json` and applies
the checked-in patches. Its MIT license is included here. llama.cpp carries the
licenses for its bundled dependencies in its own source tree.

Prebuilt Linux runtime archives also include `rotate-bits-LICENSE.txt`, the
unchanged MIT license and copyright notice for William Casarin's rotate-bits
header at the pinned llama.cpp revision. The vendored SHA-256 implementation
uses that header; its notice accompanies the compiled runtime.

The repository's MIT license covers inference code. Model weights and the vision
projector are separate downloads governed by their respective model terms.
