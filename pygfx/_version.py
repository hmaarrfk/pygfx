# -*- coding: utf-8 -*-
# This file is part of 'miniver': https://github.com/jbweston/miniver
#
from collections import namedtuple
import os

Version = namedtuple("Version", ("release", "post", "labels"))

# No public API
__all__ = []

package_root = os.path.dirname(os.path.realpath(__file__))
package_name = os.path.basename(package_root)
STATIC_VERSION_FILE = "_static_version.py"


def get_version():
    version = get_version_from_git()
    if not version:
        version = Version("unknown", None, None)
    return pep440_format(version)


def pep440_format(version_info):
    release, post, labels = version_info

    version_parts = [release]
    if post:
        if release.endswith("-post") or release.endswith(".post"):
            version_parts.append(post)
        else:  # prefer PEP440 over strict adhesion to semver
            version_parts.append(".post{}".format(post))

    if labels:
        version_parts.append("+")
        version_parts.append(".".join(labels))

    return "".join(version_parts)


def get_version_from_git():
    import subprocess

    # git describe --first-parent does not take into account tags from branches
    # that were merged-in. The '--long' flag gets us the 'post' version and
    # git hash, '--always' returns the git hash even if there are no tags.
    for opts in [["--first-parent"], []]:
        try:
            p = subprocess.Popen(
                ["git", "describe", "--long", "--always", "--tags", "--dirty"] + opts,
                cwd=package_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError:
            return
        if p.wait() == 0:
            break
    else:
        return
    if os.environ.get("PYGFX_GIT_DESCRIBE", None):
        git_describe = os.environ["PYGFX_GIT_DESCRIBE"]
    else:
        git_describe = p.communicate()[0].decode()

    description = (
        git_describe
        .lstrip("v")  # Tags can have a leading 'v', but the version should not
        .rstrip("\n")
        .rsplit("-")  # Split the latest tag, commits since tag, and hash, and dirty
    )

    try:
        release, post, git = description[:3]
    except ValueError:  # No tags, only the git hash
        # prepend 'g' to match with format returned by 'git describe'
        git = "g{}".format(*description)
        release = "nodescription"
        post = None

    labels = []
    if post == "0":
        post = None
    else:
        labels.append(git)

    if description[-1] == "dirty":
        labels.append("dirty")

    return Version(release, post, labels)


__version__ = get_version()


# The following section defines a 'get_cmdclass' function
# that can be used from setup.py. The '__version__' module
# global is used (but not modified).


def _write_version(fname):
    # This could be a hard link, so try to delete it first.  Is there any way
    # to do this atomically together with opening?
    try:
        os.remove(fname)
    except OSError:
        pass
    with open(fname, "w") as f:
        f.write(f"__version__ = '{__version__}'")


def get_cmdclass(pkg_source_path):
    from setuptools.command.build_py import build_py as build_py_orig
    from setuptools.command.sdist import sdist as sdist_orig

    class _build_py(build_py_orig):  # noqa
        def run(self):
            super().run()

            src_marker = "".join(["src", os.path.sep])

            if pkg_source_path.startswith(src_marker):
                path = pkg_source_path[len(src_marker) :]
            else:
                path = pkg_source_path
            _write_version(os.path.join(self.build_lib, path, "_version.py"))

    class _sdist(sdist_orig):  # noqa
        def make_release_tree(self, base_dir, files):
            super().make_release_tree(base_dir, files)
            _write_version(os.path.join(base_dir, pkg_source_path, "_version.py"))

    return dict(sdist=_sdist, build_py=_build_py)