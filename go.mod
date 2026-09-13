module github.com/sprout-foundry/sprout-local

go 1.25.6

require github.com/sprout-foundry/sinter v0.3.0

require golang.org/x/sys v0.45.0

require github.com/gorilla/websocket v1.5.3

require github.com/sprout-foundry/seed v1.4.0

// Local copy of sinter, pinned verbatim to v0.3.0 (plus .gitignore). Kept
// vendored so the release pipeline builds from a fixed tree; resync with
// rsync from a fresh upstream checkout when bumping the pin.
replace github.com/sprout-foundry/sinter => ./third_party/sinter
