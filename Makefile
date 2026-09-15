# sprout-local — local chat with some extra help (in-process sinter inference)

BINARY  = sprout-local
INSTALL = $(HOME)/.local/bin/$(BINARY)

UNAME_S := $(shell uname -s 2>/dev/null)

# sinter backend selection:
#   macOS (Apple Silicon)      -> MLX backend, no extra flags
#   Linux / Termux (android)   -> GGML backend: needs -tags ggml + cgo
GO_TAGS :=
ifeq ($(UNAME_S),Linux)
GO_TAGS := -tags ggml
endif

# sinter's cgo directives hardcode /usr/local/{include,lib}. On Termux the
# local libggml/libggml-base live under $PREFIX instead; point CGO there.
# (gated to Linux: a stray PREFIX on macOS must not leak -lggml into the
# MLX build; on plain Linux without $PREFIX the /usr/local defaults apply,
# i.e. libggml must be installed there for the GGML backend)
PREFIX ?= $(shell printenv PREFIX 2>/dev/null)
ifneq ($(PREFIX),)
ifeq ($(UNAME_S),Linux)
export CGO_CFLAGS := -I$(PREFIX)/include
export CGO_LDFLAGS := -L$(PREFIX)/lib -lggml -lggml-base -lm
endif
endif

GO := CGO_ENABLED=1 go

.PHONY: all build install test clean serve

all: build

build:
	$(GO) build $(GO_TAGS) -o $(BINARY) ./cmd/sprout-local

install: build
	mkdir -p $(dir $(INSTALL))
	ln -sf $(CURDIR)/$(BINARY) $(INSTALL)
	@echo "Installed → $(INSTALL)"

test:
	$(GO) test $(GO_TAGS) ./...

serve: build
	./$(BINARY) -serve

clean:
	rm -f $(BINARY)
