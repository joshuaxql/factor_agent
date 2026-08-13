.DEFAULT_GOAL := help

ifeq ($(OS),Windows_NT)
SHELL := cmd.exe
.SHELLFLAGS := /C
PY_LIBDIR = $(PY_HOME)/libs
PY_LIB = python$(subst .,,$(PY_VER))
CXXFLAGS = -O1 -mcmodel=small -fno-strict-aliasing -std=c++17 -pipe $(PY_WIN64_DEFINE)
LDFLAGS := -static-libgcc -static-libstdc++
OBJ_SUFFIX := .o
MKDIR_BUILD = if not exist "$(BUILDDIR)" mkdir "$(BUILDDIR)"
CLEAN_DIRS = if exist "$(BUILDDIR)" rmdir /S /Q "$(BUILDDIR)" & if exist "pyqlib.egg-info" rmdir /S /Q "pyqlib.egg-info" & if exist "build" rmdir /S /Q "build"
COPY_RUNTIME_DLLS = powershell -NoProfile -ExecutionPolicy Bypass -Command "$$cxx=(Get-Command '$(CXX)' -ErrorAction SilentlyContinue).Source; if ($$cxx) { $$bin=Split-Path -Parent $$cxx; foreach ($$dll in 'libwinpthread-1.dll','libgcc_s_seh-1.dll','libstdc++-6.dll') { $$src=Join-Path $$bin $$dll; if (Test-Path $$src) { Copy-Item -Force $$src '$(LIBDIR)' } } }"
CLEAN_EXT = powershell -NoProfile -ExecutionPolicy Bypass -Command "Remove-Item -Force -ErrorAction SilentlyContinue '$(LIBDIR)/*.cpp','$(LIBDIR)/*.pyd','$(LIBDIR)/*.so','$(LIBDIR)/*.o','$(LIBDIR)/*.obj','$(LIBDIR)/*.dll'"
CLEAN_PYCACHE = powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem -Path . -Directory -Recurse -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
else
SHELL := /bin/bash
PY_LIBDIR = $(PY_HOME)/lib
PY_LIB = python$(PY_VER)
CXXFLAGS := -O1 -fno-strict-aliasing -std=c++17 -fPIC -pipe
LDFLAGS :=
OBJ_SUFFIX := .o
MKDIR_BUILD = mkdir -p "$(BUILDDIR)"
COPY_RUNTIME_DLLS = true
CLEAN_DIRS = rm -rf "$(BUILDDIR)" "pyqlib.egg-info" "build"
CLEAN_EXT = rm -f "$(LIBDIR)"/*.cpp "$(LIBDIR)"/*.so "$(LIBDIR)"/*.pyd "$(LIBDIR)"/*.o "$(LIBDIR)"/*.obj "$(LIBDIR)"/*.dll
CLEAN_PYCACHE = find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
endif

PYTHON ?= python

LIBDIR := qlib/data/_libs
BUILDDIR := build/temp

PY_VER = $(shell $(PYTHON) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_HOME = $(shell $(PYTHON) -c "import sys; print(sys.base_prefix)")
PY_INCLUDE = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
NUMPY_INCLUDE = $(shell $(PYTHON) -c "import numpy; print(numpy.get_include())")
EXT_SUFFIX = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PY_WIN64_DEFINE = $(shell $(PYTHON) -c "import struct; print('-DMS_WIN64' if struct.calcsize('P') == 8 else '')")

CXX := g++

ROLLING_SRC := $(LIBDIR)/rolling.pyx
EXPANDING_SRC := $(LIBDIR)/expanding.pyx
ROLLING_CPP := $(LIBDIR)/rolling.cpp
EXPANDING_CPP := $(LIBDIR)/expanding.cpp
ROLLING_EXT = $(LIBDIR)/rolling$(EXT_SUFFIX)
EXPANDING_EXT = $(LIBDIR)/expanding$(EXT_SUFFIX)

.PHONY: help build-ext verify clean

help:
	@echo Available targets:
	@echo   make build-ext  - compile Cython extensions
	@echo   make verify     - print compiled extension paths
	@echo   make clean      - remove local build artifacts

# --- Cython: .pyx -> .cpp ---

$(ROLLING_CPP): $(ROLLING_SRC)
	@echo "[cython] $< -> $@"
	$(PYTHON) -m cython -3 --cplus "$<" -o "$@"

$(EXPANDING_CPP): $(EXPANDING_SRC)
	@echo "[cython] $< -> $@"
	$(PYTHON) -m cython -3 --cplus "$<" -o "$@"

# --- g++: .cpp -> extension module ---

build-ext: $(ROLLING_CPP) $(EXPANDING_CPP)
	@echo "[compile] $(ROLLING_CPP) -> $(ROLLING_EXT)"
	$(MKDIR_BUILD)
	$(CXX) -c $(CXXFLAGS) -I"$(PY_INCLUDE)" -I"$(NUMPY_INCLUDE)" "$(ROLLING_CPP)" -o "$(BUILDDIR)/rolling$(OBJ_SUFFIX)"
	$(CXX) -shared "$(BUILDDIR)/rolling$(OBJ_SUFFIX)" -o "$(ROLLING_EXT)" -L"$(PY_LIBDIR)" -l$(PY_LIB) $(LDFLAGS)
	@echo "[compile] $(EXPANDING_CPP) -> $(EXPANDING_EXT)"
	$(MKDIR_BUILD)
	$(CXX) -c $(CXXFLAGS) -I"$(PY_INCLUDE)" -I"$(NUMPY_INCLUDE)" "$(EXPANDING_CPP)" -o "$(BUILDDIR)/expanding$(OBJ_SUFFIX)"
	$(CXX) -shared "$(BUILDDIR)/expanding$(OBJ_SUFFIX)" -o "$(EXPANDING_EXT)" -L"$(PY_LIBDIR)" -l$(PY_LIB) $(LDFLAGS)
	$(COPY_RUNTIME_DLLS)

verify:
	$(PYTHON) -c "import os, sys, tempfile; root=os.getcwd(); os.chdir(tempfile.mkdtemp(prefix='qlib_verify_')); sys.path.insert(0, root); import qlib; import qlib.data._libs.rolling as r; import qlib.data._libs.expanding as e; print('qlib', qlib.__version__); print(r.__file__); print(e.__file__)"

clean:
	$(CLEAN_DIRS)
	$(CLEAN_EXT)
	$(CLEAN_PYCACHE)
