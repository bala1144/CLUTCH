import ctypes
import glob
import os
import sys


def _iter_library_dirs():
    candidates = []
    env_prefix = os.environ.get("CONDA_PREFIX")
    if env_prefix:
        candidates.append(os.path.join(env_prefix, "lib"))

    prefix = sys.prefix
    candidates.append(os.path.join(prefix, "lib"))

    root = os.path.dirname(os.path.dirname(prefix))
    candidates.extend(sorted(glob.glob(os.path.join(root, "envs", "*", "lib"))))

    pkg_patterns = [
        "pkgs/libegl-*/lib",
        "pkgs/libgl-*/lib",
        "pkgs/libglvnd-*/lib",
        "pkgs/libglx-*/lib",
        "pkgs/osmesa-*/lib",
        "pkgs/libxcb-*/lib",
        "pkgs/xorg-libx11-*/lib",
        "pkgs/xorg-libxext-*/lib",
        "pkgs/xorg-libxau-*/lib",
        "pkgs/xorg-libxdmcp-*/lib",
        "pkgs/xorg-libxrender-*/lib",
    ]
    for pattern in pkg_patterns:
        candidates.extend(sorted(glob.glob(os.path.join(root, pattern)), reverse=True))

    # Keep order but drop duplicates/missing paths.
    seen = set()
    for path in candidates:
        if path and path not in seen and os.path.isdir(path):
            seen.add(path)
            yield path


def _find_library(filename):
    for lib_dir in _iter_library_dirs():
        path = os.path.join(lib_dir, filename)
        if os.path.exists(path):
            return path
    return None


def _preload(filename):
    path = _find_library(filename)
    if not path:
        return None
    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    return path


def configure_pyrender_backend(preferred="egl"):
    """
    Preload the GL/X11 runtime that pyrender/pyglet need in headless mode.
    This must run before importing pyrender.
    """
    os.environ.setdefault("PYGLET_HEADLESS", "1")

    backend = preferred or os.environ.get("PYOPENGL_PLATFORM") or "egl"
    if backend == "egl":
        # Load transitive X11/GL deps before libEGL/libGL so dlopen can resolve them.
        for lib_name in [
            "libXau.so.6",
            "libXdmcp.so.6",
            "libxcb.so.1",
            "libX11.so.6",
            "libXext.so.6",
            "libXrender.so.1",
            "libGLdispatch.so.0",
            "libGLX.so.0",
            "libGL.so.1",
        ]:
            _preload(lib_name)

        egl_path = _preload("libEGL.so.1")
        if egl_path:
            os.environ["PYOPENGL_PLATFORM"] = "egl"
            os.environ.setdefault("PYOPENGL_EGL_LIBRARY", egl_path)
            return "egl"

    if backend == "osmesa" or not os.environ.get("PYOPENGL_PLATFORM"):
        osmesa_path = _preload("libOSMesa.so.8")
        if osmesa_path:
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"
            os.environ.setdefault("PYOPENGL_OSMESA_LIBRARY", osmesa_path)
            return "osmesa"

    return os.environ.get("PYOPENGL_PLATFORM", backend)
