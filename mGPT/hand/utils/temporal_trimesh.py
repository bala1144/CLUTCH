import trimesh
import numpy as np

class temporal_trimesh():
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces

    def __getitem__(self, index, scale=1.0):
        vertices = self.vertices[index] * scale
        return trimesh.Trimesh(vertices=vertices, faces=self.faces,  process=False)
    
    def get_item(self, index, colours=None):
        vertices = self.vertices[index]
        return trimesh.Trimesh(vertices=vertices, faces=self.faces, colours=colours, process=False)
    
    def __len__(self):
        return self.vertices.shape[0]

    def apply(self, function, **kwargs):
        out = []
        for i in range(len(self.vertices)):
            out.append(function(self.get_item(i), **kwargs))
        return out
    
    def apply_obj_fn(self, fn, obj_mesh, **kwargs):

        out = []
        for i in range(len(self.vertices)):
            out.append(fn(self.get_item(i), obj_mesh.get_item(i), **kwargs))
        return out
    
    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return temporal_trimesh(self.vertices * value, self.faces)
        
    def __rmul__(self, other):
        # This is for cases where the scalar comes first (e.g., 3 * Vector)
        return self.__mul__(other)
        
    