import os
import re
import time
from typing import List, Dict, Optional, Tuple
import numpy as np
from OpenGL.GL import *


VERTEX_SHADER_SOURCE = """#version 330 core
layout(location = 0) in vec2 a_pos;
out vec2 v_uv;

void main() {
    v_uv = vec2((a_pos.x + 1.0) * 0.5, (1.0 - a_pos.y) * 0.5);
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""

FALLBACK_FRAG_SOURCE = """#version 330 core
out vec4 FragColor;
in vec2 v_uv;
uniform sampler2D u_camera;

void main() {
    FragColor = texture(u_camera, v_uv);
}
"""


class ShaderProgram:
    def __init__(self, name: str, filepath: str, program_id: int):
        self.name = name
        self.filepath = filepath
        self.program_id = program_id
        self.uniform_locations: Dict[str, int] = {}
        self._cache_uniforms()

    def _cache_uniforms(self):
        """Query and cache uniform locations."""
        names = [
            "u_camera", "u_resolution", "u_time", "u_hand_box", "u_has_box",
            "u_mask_mode", "u_intensity", "u_mouse",
            "u_thumb1", "u_index1", "u_thumb2", "u_index2",
            "u_num_frozen"
        ]
        for name in names:
            loc = glGetUniformLocation(self.program_id, name)
            self.uniform_locations[name] = loc

        for i in range(8):
            self.uniform_locations[f"u_frozen_boxes[{i}]"] = glGetUniformLocation(self.program_id, f"u_frozen_boxes[{i}]")
            self.uniform_locations[f"u_frozen_is_quad[{i}]"] = glGetUniformLocation(self.program_id, f"u_frozen_is_quad[{i}]")
            for j in range(4):
                idx = i * 4 + j
                self.uniform_locations[f"u_frozen_pts[{idx}]"] = glGetUniformLocation(self.program_id, f"u_frozen_pts[{idx}]")

    def use(self):
        glUseProgram(self.program_id)

    def set_uniform_1i(self, name: str, val: int):
        loc = self.uniform_locations.get(name, -1)
        if loc != -1:
            glUniform1i(loc, val)

    def set_uniform_1f(self, name: str, val: float):
        loc = self.uniform_locations.get(name, -1)
        if loc != -1:
            glUniform1f(loc, val)

    def set_uniform_2f(self, name: str, x: float, y: float):
        loc = self.uniform_locations.get(name, -1)
        if loc != -1:
            glUniform2f(loc, x, y)

    def set_uniform_4f(self, name: str, x: float, y: float, z: float, w: float):
        loc = self.uniform_locations.get(name, -1)
        if loc != -1:
            glUniform4f(loc, x, y, z, w)

    def delete(self):
        if self.program_id:
            glDeleteProgram(self.program_id)
            self.program_id = 0


class ShaderManager:
    def __init__(self, shaders_dir: str):
        self.shaders_dir = os.path.abspath(shaders_dir)
        self.programs: List[ShaderProgram] = []
        self.active_index = 0
        self.last_error: Optional[str] = None
        self._file_mtimes: Dict[str, float] = {}

        self.vao = 0
        self.vbo = 0
        self._init_screen_quad()

        self.fallback_program: Optional[ShaderProgram] = None
        self._init_fallback_program()

        self.reload_all_shaders()

    def _init_screen_quad(self):
        quad_vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0,
        ], dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * 4, ctypes.c_void_p(0))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def _init_fallback_program(self):
        prog = self._compile_shader_from_source("fallback", VERTEX_SHADER_SOURCE, FALLBACK_FRAG_SOURCE, "")
        if prog:
            self.fallback_program = prog

    def _resolve_includes(self, source: str, base_dir: str, depth: int = 0, max_depth: int = 10, visited: Optional[set] = None) -> str:
        if depth >= max_depth:
            return source + f"\n// Error: Maximum include depth ({max_depth}) exceeded\n"

        if visited is None:
            visited = set()

        def replace_include(match):
            inc_file = match.group(1)
            inc_path = os.path.normpath(os.path.join(base_dir, inc_file))
            inc_key = os.path.normcase(inc_path)
            if inc_key in visited:
                return f"\n// Circular include detected: {inc_file}\n"

            if os.path.exists(inc_path):
                try:
                    with open(inc_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    new_visited = set(visited)
                    new_visited.add(inc_key)
                    new_base_dir = os.path.dirname(inc_path)
                    resolved_content = self._resolve_includes(
                        content, new_base_dir, depth=depth + 1, max_depth=max_depth, visited=new_visited
                    )
                    return f"\n// Begin #include \"{inc_file}\"\n" + resolved_content + f"\n// End #include \"{inc_file}\"\n"
                except Exception as e:
                    return f"\n// Error reading include {inc_file}: {e}\n"
            return f"\n// Include not found: {inc_file}\n"

        return re.sub(r'#include\s+["<]([^">]+)[">]', replace_include, source)

    def _compile_shader_from_source(self, name: str, vert_src: str, frag_src: str, filepath: str) -> Optional[ShaderProgram]:
        vert_id = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vert_id, vert_src)
        glCompileShader(vert_id)
        if not glGetShaderiv(vert_id, GL_COMPILE_STATUS):
            err = glGetShaderInfoLog(vert_id).decode("utf-8")
            glDeleteShader(vert_id)
            self.last_error = f"Vertex Shader Error ({name}): {err}"
            print(f"[Shader] {self.last_error}")
            return None

        frag_id = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(frag_id, frag_src)
        glCompileShader(frag_id)
        if not glGetShaderiv(frag_id, GL_COMPILE_STATUS):
            err = glGetShaderInfoLog(frag_id).decode("utf-8")
            glDeleteShader(vert_id)
            glDeleteShader(frag_id)
            self.last_error = f"Frag Shader Error ({name}):\n{err}"
            print(f"[Shader] {self.last_error}")
            return None

        prog_id = glCreateProgram()
        glAttachShader(prog_id, vert_id)
        glAttachShader(prog_id, frag_id)
        glLinkProgram(prog_id)

        glDeleteShader(vert_id)
        glDeleteShader(frag_id)

        if not glGetProgramiv(prog_id, GL_LINK_STATUS):
            err = glGetProgramInfoLog(prog_id).decode("utf-8")
            glDeleteProgram(prog_id)
            self.last_error = f"Link Error ({name}): {err}"
            print(f"[Shader] {self.last_error}")
            return None

        return ShaderProgram(name, filepath, prog_id)

    def compile_shader_file(self, filepath: str) -> Optional[ShaderProgram]:
        filename = os.path.basename(filepath)
        name = os.path.splitext(filename)[0]
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw_frag = f.read()
        except Exception as e:
            self.last_error = f"File read error ({filename}): {e}"
            print(f"[Shader] {self.last_error}")
            return None

        resolved_frag = self._resolve_includes(raw_frag, os.path.dirname(filepath))
        return self._compile_shader_from_source(name, VERTEX_SHADER_SOURCE, resolved_frag, filepath)

    def reload_all_shaders(self):
        if not os.path.exists(self.shaders_dir):
            os.makedirs(self.shaders_dir, exist_ok=True)

        entries = [
            os.path.join(self.shaders_dir, f) for f in os.listdir(self.shaders_dir)
            if f.endswith(".frag")
        ]
        entries.sort()

        new_programs = []
        for filepath in entries:
            prog = self.compile_shader_file(filepath)
            if prog:
                new_programs.append(prog)
                self._file_mtimes[filepath] = os.path.getmtime(filepath)

        for f in os.listdir(self.shaders_dir):
            if f.endswith(".glsl"):
                p = os.path.join(self.shaders_dir, f)
                self._file_mtimes[p] = os.path.getmtime(p)

        if new_programs:
            for p in self.programs:
                p.delete()
            self.programs = new_programs
            self.active_index = min(self.active_index, len(self.programs) - 1)
            self.last_error = None
            print(f"[Shader] Loaded {len(self.programs)} shaders successfully.")
        else:
            print("[Shader] No shaders compiled successfully. Using fallback.")

    def check_hot_reload(self) -> bool:
        reloaded = False
        glsl_changed = False
        for f in os.listdir(self.shaders_dir):
            if f.endswith(".glsl"):
                glsl_path = os.path.join(self.shaders_dir, f)
                mtime = os.path.getmtime(glsl_path)
                if mtime > self._file_mtimes.get(glsl_path, 0.0):
                    self._file_mtimes[glsl_path] = mtime
                    glsl_changed = True

        if glsl_changed:
            # Recompile all shaders if any header changed
            self.reload_all_shaders()
            return True

        # Check individual shader files
        for i, prog in enumerate(self.programs):
            if os.path.exists(prog.filepath):
                mtime = os.path.getmtime(prog.filepath)
                if mtime > self._file_mtimes.get(prog.filepath, 0.0):
                    self._file_mtimes[prog.filepath] = mtime
                    new_prog = self.compile_shader_file(prog.filepath)
                    if new_prog:
                        prog.delete()
                        self.programs[i] = new_prog
                        self.last_error = None
                        print(f"[Shader] Hot-reloaded: {prog.name}")
                        reloaded = True
                    else:
                        print(f"[Shader] Hot-reload failed for {prog.name}. Keeping previous version.")

        # Check if new .frag files were added to directory
        current_files = set(
            os.path.join(self.shaders_dir, f) for f in os.listdir(self.shaders_dir)
            if f.endswith(".frag")
        )
        loaded_files = set(p.filepath for p in self.programs)
        if current_files != loaded_files:
            self.reload_all_shaders()
            reloaded = True

        return reloaded

    @property
    def current_program(self) -> ShaderProgram:
        if self.programs:
            return self.programs[self.active_index]
        return self.fallback_program

    @property
    def active_shader_name(self) -> str:
        if self.programs:
            return self.programs[self.active_index].name
        return "Fallback Passthrough"

    @property
    def shader_count(self) -> int:
        return len(self.programs)

    def next_shader(self):
        if self.programs:
            self.active_index = (self.active_index + 1) % len(self.programs)

    def prev_shader(self):
        if self.programs:
            self.active_index = (self.active_index - 1) % len(self.programs)

    def select_shader(self, index: int):
        if 0 <= index < len(self.programs):
            self.active_index = index

    def render_quad(self, texture_id: int, width: int, height: int, elapsed_time: float,
                    hand_box: Tuple[float, float, float, float], has_box: float,
                    mask_mode: int, intensity: float,
                    mouse_uv: Tuple[float, float],
                    thumb1: Tuple[float, float], index1: Tuple[float, float],
                    thumb2: Tuple[float, float], index2: Tuple[float, float],
                    frozen_shapes: Optional[List[Any]] = None):
        prog = self.current_program
        if not prog or not prog.program_id:
            return

        prog.use()

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        prog.set_uniform_1i("u_camera", 0)

        prog.set_uniform_2f("u_resolution", float(width), float(height))
        prog.set_uniform_1f("u_time", float(elapsed_time))
        prog.set_uniform_4f("u_hand_box", float(hand_box[0]), float(hand_box[1]), float(hand_box[2]), float(hand_box[3]))
        prog.set_uniform_1f("u_has_box", float(has_box))
        prog.set_uniform_1i("u_mask_mode", int(mask_mode))
        prog.set_uniform_1f("u_intensity", float(intensity))
        prog.set_uniform_2f("u_mouse", float(mouse_uv[0]), float(mouse_uv[1]))
        prog.set_uniform_2f("u_thumb1", float(thumb1[0]), float(thumb1[1]))
        prog.set_uniform_2f("u_index1", float(index1[0]), float(index1[1]))
        prog.set_uniform_2f("u_thumb2", float(thumb2[0]), float(thumb2[1]))
        prog.set_uniform_2f("u_index2", float(index2[0]), float(index2[1]))

        num_frozen = min(len(frozen_shapes), 8) if frozen_shapes else 0
        prog.set_uniform_1i("u_num_frozen", num_frozen)

        if frozen_shapes:
            for i in range(num_frozen):
                shape = frozen_shapes[i]
                is_quad = getattr(shape, "is_quad", False)
                prog.set_uniform_1i(f"u_frozen_is_quad[{i}]", 1 if is_quad else 0)
                box = getattr(shape, "box", (0.0, 0.0, 0.0, 0.0))
                prog.set_uniform_4f(f"u_frozen_boxes[{i}]", float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                pts = getattr(shape, "pts", [])
                for j in range(4):
                    idx = i * 4 + j
                    pt = pts[j] if j < len(pts) else (0.0, 0.0)
                    prog.set_uniform_2f(f"u_frozen_pts[{idx}]", float(pt[0]), float(pt[1]))

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)

    def cleanup(self):
        for p in self.programs:
            p.delete()
        if self.fallback_program:
            self.fallback_program.delete()
            self.fallback_program = None
        if self.vbo:
            glDeleteBuffers(1, [self.vbo])
            self.vbo = 0
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
            self.vao = 0
