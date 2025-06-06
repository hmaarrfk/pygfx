import pygfx
from pygfx.renderers.wgpu.shader import BaseShader
from pygfx.renderers.wgpu import Binding, register_wgsl_loader
from pygfx.utils import array_from_shadertype
from pygfx import Buffer
from pytest import raises
import jinja2
import numpy as np


def get_bindings_code(shader):
    return shader._binding_definitions.get_code()


class LocalBaseShader(BaseShader):
    def __init__(self, **kwargs):
        super().__init__(None, **kwargs)  # No world object


def test_templating():
    class MyShader(LocalBaseShader):
        def get_code(self):
            return """
            $$ if foo
            x = {{bar}}
            $$ else
            x = {{bar + 1}}
            $$ endif
            """

    # Missing variables
    shader = MyShader(foo=True)
    with raises(ValueError):
        shader.generate_wgsl()

    # Fill in value
    shader["bar"] = 42
    assert shader["bar"] == 42
    assert shader.generate_wgsl().strip() == "x = 42"

    # Can also specify when generating
    assert shader.generate_wgsl(foo=False).strip() == "x = 43"

    # But these only apply for that call
    assert shader.generate_wgsl().strip() == "x = 42"

    # Inline block notation
    class MyShader(LocalBaseShader):
        def get_code(self):
            return """
            {$ if foo $} 1 {$ else $} 2 {$ endif $}
            """

    shader = MyShader(foo=True)
    assert shader.generate_wgsl().strip() == "1"
    assert shader.generate_wgsl(foo=False).strip() == "2"


def test_logic_beyond_templating():
    class MyShader(LocalBaseShader):
        def get_code(self):
            if self["foo"]:
                return "x = {{bar + 1}}"
            else:
                return "x = {{bar}}"

    shader = MyShader(foo=False, bar=24)

    assert shader.generate_wgsl().strip() == "x = 24"
    shader["foo"] = True
    assert shader.generate_wgsl().strip() == "x = 25"

    assert shader.generate_wgsl(foo=False).strip() == "x = 24"
    assert shader.generate_wgsl().strip() == "x = 25"

    assert shader.generate_wgsl(bar=1).strip() == "x = 2"
    assert shader.generate_wgsl().strip() == "x = 25"


def test_uniform_definitions():
    class MyShader(LocalBaseShader):
        def get_code(self):
            return ""

        def clear_bindings(self):
            self._binding_definitions._uniform_struct_names.clear()
            self._binding_definitions._typedefs.clear()

    shader = MyShader()

    # Make it look like we're inside get_bindings ...
    shader._template_vars_current = shader._template_vars_bindings

    # Fails
    with raises(TypeError):  # Not a valid struct type
        shader.define_binding(0, 0, Binding("zz", "buffer/uniform", "not a struct"))
    with raises(TypeError):  # Not a valid struct type
        shader.define_binding(
            0, 0, Binding("zz", "buffer/uniform", np.array([1]).dtype)
        )

    # Test simple scalars
    struct = dict(foo="f4", bar="i4")
    shader.define_binding(0, 0, Binding("zz", "buffer/uniform", struct))
    assert (
        get_bindings_code(shader).strip()
        == """
        struct Struct_u_1 {
            foo: f32,
            bar: i32,
        };

        @group(0) @binding(0)
        var<uniform> zz: Struct_u_1;
    """.strip()
    )

    # Test vec
    struct = dict(foo="4xf4", bar="2xi4")
    shader.clear_bindings()
    shader.define_binding(0, 0, Binding("zz", "buffer/uniform", struct))
    assert (
        get_bindings_code(shader).strip()
        == """
        struct Struct_u_1 {
            foo: vec4<f32>,
            bar: vec2<i32>,
        };

        @group(0) @binding(0)
        var<uniform> zz: Struct_u_1;
    """.strip()
    )

    # Test mat
    struct = dict(foo="4x4xf4", bar="3x2xi4")
    shader.clear_bindings()
    shader.define_binding(0, 0, Binding("zz", "buffer/uniform", struct))
    assert (
        get_bindings_code(shader).strip()
        == """
        struct Struct_u_1 {
            foo: mat4x4<f32>,
            bar: mat3x2<i32>,
        };

        @group(0) @binding(0)
        var<uniform> zz: Struct_u_1;
    """.strip()
    )

    # Test array
    struct = dict(foo="4x4xf4", bar="3x2xi4")
    shader.clear_bindings()

    shader.define_binding(
        0,
        0,
        Binding(
            "zz",
            "buffer/uniform",
            Buffer(array_from_shadertype(struct, 3)),
            structname="Struct_Foo",
        ),
    )
    assert (
        get_bindings_code(shader).strip()
        == """
        struct Struct_Foo {
            foo: mat4x4<f32>,
            bar: mat3x2<i32>,
        };

        @group(0) @binding(0)
        var<uniform> zz: array<Struct_Foo, 3>;
    """.strip()
    )


def test_custom_wgsl_loaders():
    class MyLoader(jinja2.BaseLoader):
        def get_source(self, environment, x):
            return f"// loader: {x}", None, None

    my_loader = MyLoader()

    def my_func(x):
        return f"// func: {x}"

    my_dict = {"foo.wgsl": "// dict: foo.wgsl"}

    register_wgsl_loader("testloader1", my_loader)
    register_wgsl_loader("testloader2", my_func)
    register_wgsl_loader("testloader3", my_dict)

    with raises(RuntimeError):
        register_wgsl_loader("testloader3", {})

    code1 = """
    {$ include 'testloader1.foo.wgsl' $}
    {$ include 'testloader2.foo.wgsl' $}
    {$ include 'testloader3.foo.wgsl' $}
    """

    code2 = """
    // loader: foo.wgsl
    // func: foo.wgsl
    // dict: foo.wgsl
    """

    code3 = pygfx.renderers.wgpu.shader.templating.apply_templating(code1)
    assert code3 == code2


if __name__ == "__main__":
    test_templating()
    test_logic_beyond_templating()
    test_uniform_definitions()
    test_custom_wgsl_loaders()
