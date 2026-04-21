import numpy as np
import trimesh
import os
from mGPT.hand.visualizer.pyrender_runtime import configure_pyrender_backend

configure_pyrender_backend()

import pyglet
pyglet.options["headless"] = True

import pyrender
from pyrender.light import DirectionalLight
from pyrender.node import Node
from pyrender.constants import RenderFlags

if os.environ.get("PYOPENGL_PLATFORM") == "egl":
    from pyrender.platforms import egl as pyrender_egl

    def _safe_get_device_by_index(device_id):
        devices = pyrender_egl.query_devices()
        if not devices or device_id < 0 or device_id >= len(devices):
            return pyrender_egl.EGLDevice(None)
        return devices[device_id]

    pyrender_egl.get_device_by_index = _safe_get_device_by_index

from PIL import Image, ImageDraw, ImageFont
# from mdm_grab.utils.transformations import euler

from glob import glob
import tqdm
import cv2

def euler(rots, order='xyz', units='deg'):

    rots = np.asarray(rots)
    single_val = False if len(rots.shape)>1 else True
    rots = rots.reshape(-1,3)
    rotmats = []

    for xyz in rots:
        if units == 'deg':
            xyz = np.radians(xyz)
        r = np.eye(3)
        for theta, axis in zip(xyz,order):
            c = np.cos(theta)
            s = np.sin(theta)
            if axis=='x':
                r = np.dot(np.array([[1,0,0],[0,c,-s],[0,s,c]]), r)
            if axis=='y':
                r = np.dot(np.array([[c,0,s],[0,1,0],[-s,0,c]]), r)
            if axis=='z':
                r = np.dot(np.array([[c,-s,0],[s,c,0],[0,0,1]]), r)
        rotmats.append(r)
    rotmats = np.stack(rotmats).astype(np.float32)
    if single_val:
        return rotmats[0]
    else:
        return rotmats
        
def grab_mesh_viewer(dataset="GRAB", camera_pose=None):

    if dataset == "amass":
        mv = MeshViewer(width=1600, height=1200, offscreen=True)
    else:
        mv = MeshViewer(width=1600, height=1200, offscreen=True)

    # # set the camera pose
    # camera_pose = np.eye(4)
    # camera_pose[:3, :3] = euler([80, -15, 0], 'xzx')
    # camera_pose[:3, 3] = np.array([-.5, -1.4, 1.5])
    # mv.update_camera_pose(camera_pose)

    if camera_pose is None:
        if dataset =="GRAB":
            camera_pose = np.array([
                [ 1., -0.,  0.,  0.],
                [ 0.,  0., -1., -1.8],
                [ 0.,  1.,  0.,  0.],
                [ 0.,  0.,  0.,  1.]])

        elif "egovid5M" in dataset:
            camera_pose = np.array([
                [ 1., -0.,  0.,  0.],
                [ 0.,  0., -1., -1.2],
                [ 0.,  1.,  0.,  0.],
                [ 0.,  0.,  0.,  1.]])
            
        elif "amass" in dataset:
            camera_pose = np.array([
            [ 0.,   0., 1., 3],
            [ 0.,  1.,  0.,  1],
            [ -1.,  0,   0.,  0.],
            [ 0.,  0.,  0.,  1.]])
            
    mv.update_camera_pose(camera_pose)

    ### addede to handle the lighting for the AMASS dataset
    if dataset == "amass":
        mv.update_light(camera_pose)

    return mv

class Mesh(trimesh.Trimesh):

    def __init__(self,
                 filename=None,
                 vertices=None,
                 faces=None,
                 vc=None,
                 fc=None,
                 vscale=None,
                 process = False,
                 visual = None,
                 wireframe=False,
                 smooth = False,
                 **kwargs):

        self.wireframe = wireframe
        self.smooth = smooth

        if filename is not None:
            mesh = trimesh.load(filename, process = process)
            vertices = mesh.vertices
            faces= mesh.faces
            visual = mesh.visual
        if vscale is not None:
            vertices = vertices*vscale

        if faces is None:
            mesh = points2sphere(vertices)
            vertices = mesh.vertices
            faces = mesh.faces
            visual = mesh.visual

        super(Mesh, self).__init__(vertices=vertices, faces=faces, process=process, visual=visual)

        if vc is not None:
            self.set_vertex_colors(vc)
        if fc is not None:
            self.set_face_colors(fc)

    def rot_verts(self, vertices, rxyz):
        return np.array(vertices * rxyz.T)

    def colors_like(self,color, array, ids):

        color = np.array(color)

        if color.max() <= 1.:
            color = color * 255
        color = color.astype(np.int8)

        n_color = color.shape[0]
        n_ids = ids.shape[0]

        new_color = np.array(array)
        if n_color <= 4:
            new_color[ids, :n_color] = np.repeat(color[np.newaxis], n_ids, axis=0)
        else:
            new_color[ids, :] = color

        return new_color

    def set_vertex_colors(self,vc, vertex_ids = None):

        all_ids = np.arange(self.vertices.shape[0])
        if vertex_ids is None:
            vertex_ids = all_ids

        vertex_ids = all_ids[vertex_ids]
        new_vc = self.colors_like(vc, self.visual.vertex_colors, vertex_ids)
        self.visual.vertex_colors[:] = new_vc

    def set_face_colors(self,fc, face_ids = None):

        if face_ids is None:
            face_ids = np.arange(self.faces.shape[0])

        new_fc = self.colors_like(fc, self.visual.face_colors, face_ids)
        self.visual.face_colors[:] = new_fc

    @staticmethod
    def concatenate_meshes(meshes):
        return trimesh.util.concatenate(meshes)

class MeshViewer(object):

    def __init__(self,
                 width=800,
                 height=800,
                 bg_color = [1.0, 1.0, 1.0, 1.0],
                 offscreen = False,
                 registered_keys=None):
        super(MeshViewer, self).__init__()

        if registered_keys is None:
            registered_keys = dict()

        self.bg_color = bg_color
        self.offscreen = offscreen
        self.scene = pyrender.Scene(bg_color=bg_color,
                                    ambient_light=(0.3, 0.3, 0.3),
                                    name = 'scene')

        self.aspect_ratio = float(width) / height
        pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=self.aspect_ratio)
        self.pc = pc
        camera_pose = np.eye(4)
        camera_pose[:3,:3] = euler([80,-15,0], 'xzx')
        camera_pose[:3, 3] = np.array([-.5, -2., 1.5])# original
        self.cam = pyrender.Node(name = 'camera', camera=pc, matrix=camera_pose)

        self.scene.add_node(self.cam)

        if self.offscreen:
            light = Node(light=DirectionalLight(color=np.ones(3), intensity=3.0),
                          matrix=camera_pose)
            self.scene.add_node(light)
            self.light_node = light
            self.viewer = pyrender.OffscreenRenderer(width, height)
        else:
            self.viewer = pyrender.Viewer(self.scene,
                                          use_raymond_lighting=True,
                                          viewport_size=(width, height),
                                          cull_faces=False,
                                          run_in_thread=True,
                                          registered_keys=registered_keys)

        for i, node in enumerate(self.scene.get_nodes()):
            if node.name is None:
                node.name = 'Req%d'%i

    def is_active(self):
        return self.viewer.is_active

    def close_viewer(self):
        if self.viewer.is_active:
            self.viewer.close_external()

    def set_background_color(self, bg_color=[1., 1., 1.]):
        self.scene.bg_color = bg_color

    def to_pymesh(self, mesh):

        wireframe = mesh.wireframe if hasattr(mesh, 'wireframe') else False
        smooth = mesh.smooth if hasattr(mesh, 'smooth') else False
        return  pyrender.Mesh.from_trimesh(mesh, wireframe=wireframe, smooth=smooth)


    def update_camera_pose(self, pose, cam=None):
        if cam is not None:
            self.scene.remove_node(self.cam)
            self.cam = pyrender.Node(name = 'camera', camera=cam, matrix=pose)
            self.scene.add_node(self.cam)
            
        elif self.offscreen:
            self.scene.set_pose(self.cam, pose=pose)
        else:
            self.viewer._default_camera_pose[:] = pose

    def update_light(self, pose):
        self.scene.set_pose(self.light_node, pose=pose)

    def _create_raymond_lights(self):
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(
                pyrender.Node(
                    light=pyrender.DirectionalLight(color=np.ones(3),
                                                    intensity=1.0),
                    matrix=matrix
                ))

        return nodes


    def set_meshes(self, meshes =[], set_type = 'static'):

        if not self.offscreen:
            self.viewer.render_lock.acquire()

        for node in self.scene.get_nodes():
            if node.name is None:
                continue
            if 'static' in set_type and 'mesh' in node.name:
                self.scene.remove_node(node)
            elif 'dynamic' in node.name:
                self.scene.remove_node(node)

        for i, mesh in enumerate(meshes):
            mesh = self.to_pymesh(mesh)
            self.scene.add(mesh, name='%s_mesh_%d'%(set_type,i))

        if not self.offscreen:
            self.viewer.render_lock.release()

    def set_static_meshes(self, meshes =[]):
        self.set_meshes(meshes=meshes, set_type='static')

    def set_dynamic_meshes(self, meshes =[]):
        self.set_meshes(meshes=meshes, set_type='dynamic')

    def save_snapshot(self, save_path, text=None):
        if not self.offscreen:
            print('We do not support rendering in Interactive mode!')
            return
        color, depth = self.viewer.render(self.scene)
        img = Image.fromarray(color)

        if text is not None:
            img = self.add_text(img, text)
        img.save(save_path)

            # width, height = img.size
            # draw = ImageDraw.Draw(img)
            # # font = ImageFont.truetype("arial.ttf", 16)
            # font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
            # # text_width, text_height = draw.textsize(text, font)
            # # 
            # # position = ((width - text_width) // 2, 10)  # Center horizontally and 10 pixels from the top
            
            # bbox = draw.textbbox((0, 0), text, font=font)
            # text_width = bbox[2] - bbox[0]
            # text_height = bbox[3] - bbox[1]

            # # Calculate position
            # position = ((width - text_width) // 2, (height-(text_height+100)))  # Center horizontally and 10 pixels from the top

            # # Create a semi-transparent black background
            # padding = 5  # Padding around the text
            # bg_bbox = (
            #     position[0] - padding,
            #     position[1] - padding,
            #     position[0] + text_width + padding,
            #     position[1] + text_height + padding
            # )
            # # draw.rectangle(bg_bbox, fill=(0, 0, 0, 200))  # Semi-transparent black
            # draw.text(position, text, (0, 0, 200), font=font, stroke_width=1)

        # img.save(save_path)
    def get_snapshot(self,  text=None):
        if not self.offscreen:
            print('We do not support rendering in Interactive mode!')
            return
        color, depth = self.viewer.render(self.scene)
        img = Image.fromarray(color)

        if text is not None:
            img = self.add_text(img, text)
        
        return np.array(img) 
    
    def get_snapshot_from_different_cameras(self, cameras, text=None, alpha_ch=False):
        
        flags=RenderFlags.NONE
        if alpha_ch:
            flags=RenderFlags.RGBA
        
        images = []
        for cam in cameras:
            self.update_camera_pose(cam)
            color, depth = self.viewer.render(self.scene, flags=flags)
            img = Image.fromarray(color)
            if text is not None:
                img = self.add_text(img, text)
            images.append(img)

        images = np.concatenate(images, axis=1)  # axis=1 stacks them side by side
        return Image.fromarray(images)
    
    def add_text(self,img, text, position=None, colour=(0, 0, 200), font_size=40):
            width, height = img.size
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            # text_width, text_height = draw.textsize(text, font)
            # 
            # position = ((width - text_width) // 2, 10)  # Center horizontally and 10 pixels from the top
            
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Calculate position
            if position is None:
                position = ((width - text_width) // 2, (height-(text_height+100)))  # Center horizontally and 10 pixels from the top

            # # Create a semi-transparent black background
            # padding = 5  # Padding around the text
            # bg_bbox = (
            #     position[0] - padding,
            #     position[1] - padding,
            #     position[0] + text_width + padding,
            #     position[1] + text_height + padding
            # )            
            draw.text(position, text, colour, font=font, stroke_width=1)

            return img

    def dump_snapshot(self, image, save_path, text=None):
        img = Image.fromarray(image)
        if text is not None:
            img = self.add_text(img, text)
        img.save(save_path)
    
    def save_scene_as_obj(self, save_path, all_mesh):

        combined = trimesh.util.concatenate(all_mesh)
        combined.export(save_path)

    def _center_cam(self, meshes):
        all_verts = [m.bounds for m in meshes]
        for node in self.scene.get_nodes():
            if node.name is None:
                continue
            if 'mesh' in node.name:
                all_verts.append(node.mesh.bounds)

        all_verts = np.vstack(all_verts)
        maxim = np.max(all_verts, axis=0)
        minim = np.min(all_verts, axis=0)
        center = (maxim + minim)/2.

        maximum = np.max(np.abs(all_verts -center), axis=0)

        max_x = maximum[0]
        max_y = maximum[1]

        tnh = np.tan(np.pi / 6.0)
        z_y = max_y / tnh
        z_x = max_x / (tnh*self.aspect_ratio)

        pose = np.eye(4)
        pose[2, 3] = max(z_y, z_x) + .1 + maxim[2]
        pose[:2, 3] = center[:2]

        # self.viewer.render_lock.acquire()
        # self.viewer._trackball._target = center
 
        # self.viewer._trackball._target = center
        # self.viewer._default_camera_pose[:] = pose
        self.update_camera_pose(pose)

        return pose
        # self.viewer.render_lock.release()

    def save_as_gif(self, render_path, fps=1, out_path=None):
    
        ## has issues with colour encoding need to fix it
        if out_path is None:
            out_path = render_path+".gif"


        ffmpeg_cmd = [
        'ffmpeg',
        '-v', 'error',
        '-y',
        '-framerate', str(fps),
        # '-pattern_type', 'glob',
        '-i', os.path.join(render_path, '%d.png'),  # Adjust this if you have different image formats
        # '-vf', 'scale=500:-1:flags=lanczos:[x];[x][1:v]paletteuse',
        '-vf', '[0:v]fps=15,scale=320:-1:flags=lanczos[x];[x][1:v]paletteus',
        '-loop', '1',
        out_path
        ]
        print("Command to run for gifs", " ".join(ffmpeg_cmd))
        os.system(" ".join(ffmpeg_cmd))
        print("Saving gifs to: ", out_path)

    def save_as_mp4(self, render_path, fps=1, out_path=None):
        ## load the imagees in the path

        if out_path is None:
            out_path = render_path+".mp4"

        seq_name = render_path.split("/")[-1]
        ffmpeg_cmd = [
        'ffmpeg',
        '-v', 'error',
        '-y',
        '-framerate', str(fps),
        '-i', os.path.join(render_path, '%d.png'),  # Adjust this if 
        # '-vf', f"drawtext=fontfile=/path/to/font.ttf:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=10:text='{seq_name}'",
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        out_path
        ]
        print("Command to run for mp4", " ".join(ffmpeg_cmd))
        os.system(" ".join(ffmpeg_cmd))
        print("Saving mp4 to: ", out_path)

        print("Removing frames from: ", render_path)
        os.system(f"rm -rf {render_path}")

        return 

    def visualize_hands_with_gt(self, out_dir, seq_name, pred, gt, faces, skip_frame=1, fps=8, text=None):

        if text is None:
            text = seq_name
    
        T = pred.shape[0]
        for frame in tqdm.tqdm(np.arange(0,T, skip_frame), desc="rendering frames"):
            all_mesh = []
            all_mesh.append(Mesh(vertices=gt[frame], faces=faces, vc=colors["pink"], smooth=True))
            all_mesh.append(Mesh(vertices=pred[frame], faces=faces, vc=colors["yellow"], smooth=True))
            self.set_static_meshes(all_mesh)

            seq_render_path = makepath(os.path.join(out_dir, seq_name, str(frame) + '.png'),  isfile=True)
            self.save_snapshot(seq_render_path, text=text)

        if T > 1:
            self.save_as_mp4(render_path=os.path.join(out_dir, seq_name), fps=fps)

        return os.path.join(out_dir, seq_name+".mp4")

    def _error_to_heatmap_colors(self, errors, min_err, max_err):
        errors = np.asarray(errors)
        denom = max(max_err - min_err, 1e-8)
        vals = np.clip((errors - min_err) / denom, 0.0, 1.0)

        colors_map = np.zeros((vals.shape[0], 4), dtype=np.uint8)
        colors_map[:, 3] = 255

        seg1 = vals <= 0.33
        seg2 = (vals > 0.33) & (vals <= 0.66)
        seg3 = vals > 0.66

        t1 = vals[seg1] / 0.33
        colors_map[seg1, 0] = 0
        colors_map[seg1, 1] = (255 * t1).astype(np.uint8)
        colors_map[seg1, 2] = 255

        t2 = (vals[seg2] - 0.33) / 0.33
        colors_map[seg2, 0] = (255 * t2).astype(np.uint8)
        colors_map[seg2, 1] = 255
        colors_map[seg2, 2] = (255 * (1.0 - t2)).astype(np.uint8)

        t3 = (vals[seg3] - 0.66) / 0.34
        colors_map[seg3, 0] = 255
        colors_map[seg3, 1] = (255 * (1.0 - t3)).astype(np.uint8)
        colors_map[seg3, 2] = 0

        return colors_map

    def _render_rgba(self, text=None):
        """Render scene with RGBA (transparent background)."""
        flags = RenderFlags.RGBA
        color, depth = self.viewer.render(self.scene, flags=flags)
        img = Image.fromarray(color, mode='RGBA')
        if text is not None:
            img = self.add_text(img, text)
        return np.array(img)

    def _compute_alpha_bbox(self, img_rgba):
        """Compute bounding box of non-transparent pixels."""
        alpha = img_rgba[:, :, 3]
        rows = np.any(alpha > 0, axis=1)
        cols = np.any(alpha > 0, axis=0)
        if not rows.any() or not cols.any():
            return 0, img_rgba.shape[1], 0, img_rgba.shape[0]
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return x_min, x_max + 1, y_min, y_max + 1

    def _crop_with_buffer(self, img_rgba, content_bbox, pad_w, pad_h):
        """Crop image around content with symmetric buffer padding.

        Args:
            img_rgba: Input RGBA image
            content_bbox: (x_min, x_max, y_min, y_max) of content in image
            pad_w: Horizontal buffer on each side
            pad_h: Vertical buffer on each side

        Returns:
            Cropped image with guaranteed symmetric padding around content.
        """
        img_h, img_w = img_rgba.shape[:2]
        cx_min, cx_max, cy_min, cy_max = content_bbox

        # Content dimensions
        content_w = cx_max - cx_min
        content_h = cy_max - cy_min

        # Final output dimensions (content + padding on both sides)
        out_w = content_w + 2 * pad_w
        out_h = content_h + 2 * pad_h

        # Create output canvas (transparent)
        output = np.zeros((out_h, out_w, 4), dtype=np.uint8)

        # Calculate what part of the original image we can copy
        # Source region in original image (clamped to image bounds)
        src_x_min = max(0, cx_min - pad_w)
        src_x_max = min(img_w, cx_max + pad_w)
        src_y_min = max(0, cy_min - pad_h)
        src_y_max = min(img_h, cy_max + pad_h)

        # Destination region in output (accounting for clamping)
        dst_x_min = pad_w - (cx_min - src_x_min)
        dst_y_min = pad_h - (cy_min - src_y_min)
        dst_x_max = dst_x_min + (src_x_max - src_x_min)
        dst_y_max = dst_y_min + (src_y_max - src_y_min)

        # Copy the region
        output[dst_y_min:dst_y_max, dst_x_min:dst_x_max] = img_rgba[src_y_min:src_y_max, src_x_min:src_x_max]

        return output

    def _stack_images_horizontal(self, images, labels=None, padding=10, label_height=30):
        """Stack images horizontally with same height, adding labels above each."""
        if not images:
            return None

        # Find max height
        max_h = max(img.shape[0] for img in images)

        # Pad each image to max height (center vertically) and add to list
        padded_images = []
        for i, img in enumerate(images):
            h, w = img.shape[:2]

            # Create padded image with white background
            total_h = max_h + label_height
            padded = np.ones((total_h, w, 4), dtype=np.uint8) * 255
            padded[:, :, 3] = 255  # Fully opaque white background

            # Add label if provided
            if labels and i < len(labels):
                label_img = Image.new('RGBA', (w, label_height), (255, 255, 255, 255))
                draw = ImageDraw.Draw(label_img)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
                except:
                    font = ImageFont.load_default()
                bbox = draw.textbbox((0, 0), labels[i], font=font)
                text_w = bbox[2] - bbox[0]
                text_x = (w - text_w) // 2
                draw.text((text_x, 5), labels[i], fill=(0, 0, 0, 255), font=font)
                padded[:label_height, :] = np.array(label_img)

            # Center the image vertically below the label
            y_offset = label_height + (max_h - h) // 2
            padded[y_offset:y_offset + h, :] = img

            padded_images.append(padded)

        # Add padding between images
        result_parts = []
        for i, img in enumerate(padded_images):
            result_parts.append(img)
            if i < len(padded_images) - 1:
                # Add vertical padding strip
                pad_strip = np.ones((img.shape[0], padding, 4), dtype=np.uint8) * 255
                pad_strip[:, :, 3] = 255
                result_parts.append(pad_strip)

        return np.concatenate(result_parts, axis=1)

    def visulize_smpl_body(self, out_dir, seq_name, pred, gt, faces, skip_frame=1, fps=8, text=None,
                           bbox_pad_w=20, bbox_pad_h=100, heatmap_err_min=0.0, heatmap_err_max=15.0):
        """
        Render GT, Pred, Pred+Heatmap, and Overlay views side by side.

        Each view is rendered separately with RGBA, cropped based on content,
        and stacked horizontally with the same height.

        Args:
            bbox_pad_w: Horizontal padding (pixels) to add around cropped content. Default 20.
            bbox_pad_h: Vertical padding (pixels) to add around cropped content. Default 20.
            heatmap_err_min: Minimum error value in cm for heatmap color mapping. Default 0.0 cm.
            heatmap_err_max: Maximum error value in cm for heatmap color mapping. Default 5.0 cm.
                             Errors above this saturate to red.
        """
        if text is None:
            text = seq_name

        # Set transparent background for RGBA rendering
        original_bg = self.scene.bg_color
        self.scene.bg_color = [0.0, 0.0, 0.0, 0.0]

        T = pred.shape[0]
        # Compute per-vertex errors in cm (SMPL-X outputs vertices in meters)
        errors = np.linalg.norm(pred - gt, axis=-1) * 100.0
        min_err = heatmap_err_min
        max_err = heatmap_err_max

        # First pass: accumulate global bounding boxes for each view type
        frame_indices = list(range(0, T, skip_frame))
        view_bboxes = {'gt': [], 'pred': [], 'heatmap': [], 'overlay': []}

        for frame in tqdm.tqdm(frame_indices, desc="Computing bounding boxes"):
            err_colors = self._error_to_heatmap_colors(errors[frame], min_err, max_err)

            # Render GT (use vertices directly, no centering - same as visualize_hands_with_gt)
            self.set_static_meshes([Mesh(vertices=gt[frame], faces=faces, vc=colors["pink"], smooth=True)])
            img_gt = self._render_rgba()
            view_bboxes['gt'].append(self._compute_alpha_bbox(img_gt))

            # Render Pred
            self.set_static_meshes([Mesh(vertices=pred[frame], faces=faces, vc=colors["yellow"], smooth=True)])
            img_pred = self._render_rgba()
            view_bboxes['pred'].append(self._compute_alpha_bbox(img_pred))

            # Render Heatmap
            self.set_static_meshes([Mesh(vertices=pred[frame], faces=faces, vc=err_colors, smooth=True)])
            img_heatmap = self._render_rgba()
            view_bboxes['heatmap'].append(self._compute_alpha_bbox(img_heatmap))

            # Render Overlay (GT + Pred)
            self.set_static_meshes([
                Mesh(vertices=gt[frame], faces=faces, vc=colors["pink"], smooth=True),
                Mesh(vertices=pred[frame], faces=faces, vc=colors["yellow"], smooth=True),
            ])
            img_overlay = self._render_rgba()
            view_bboxes['overlay'].append(self._compute_alpha_bbox(img_overlay))

        # Compute single global content bounding box across ALL views and ALL frames
        all_bboxes = []
        for view_name, bboxes in view_bboxes.items():
            all_bboxes.extend(bboxes)

        global_x_min = min(b[0] for b in all_bboxes)
        global_x_max = max(b[1] for b in all_bboxes)
        global_y_min = min(b[2] for b in all_bboxes)
        global_y_max = max(b[3] for b in all_bboxes)

        # Single unified content bounding box for all views
        unified_content_bbox = (global_x_min, global_x_max, global_y_min, global_y_max)

        # Second pass: render and save frames
        for idx, frame in enumerate(tqdm.tqdm(frame_indices, desc="Rendering frames")):
            err_colors = self._error_to_heatmap_colors(errors[frame], min_err, max_err)

            rendered_views = []
            labels = ['GT', 'Pred', 'Heatmap', 'Overlay']

            # Render GT
            self.set_static_meshes([Mesh(vertices=gt[frame], faces=faces, vc=colors["pink"], smooth=True)])
            img_gt = self._render_rgba()
            cropped_gt = self._crop_with_buffer(img_gt, unified_content_bbox, bbox_pad_w, bbox_pad_h)
            rendered_views.append(cropped_gt)

            # Render Pred
            self.set_static_meshes([Mesh(vertices=pred[frame], faces=faces, vc=colors["yellow"], smooth=True)])
            img_pred = self._render_rgba()
            cropped_pred = self._crop_with_buffer(img_pred, unified_content_bbox, bbox_pad_w, bbox_pad_h)
            rendered_views.append(cropped_pred)

            # Render Heatmap
            self.set_static_meshes([Mesh(vertices=pred[frame], faces=faces, vc=err_colors, smooth=True)])
            img_heatmap = self._render_rgba()
            cropped_heatmap = self._crop_with_buffer(img_heatmap, unified_content_bbox, bbox_pad_w, bbox_pad_h)
            rendered_views.append(cropped_heatmap)

            # Render Overlay (GT + Pred)
            self.set_static_meshes([
                Mesh(vertices=gt[frame], faces=faces, vc=colors["pink"], smooth=True),
                Mesh(vertices=pred[frame], faces=faces, vc=colors["yellow"], smooth=True),
            ])
            img_overlay = self._render_rgba()
            cropped_overlay = self._crop_with_buffer(img_overlay, unified_content_bbox, bbox_pad_w, bbox_pad_h)
            rendered_views.append(cropped_overlay)

            # Stack all views horizontally
            combined = self._stack_images_horizontal(rendered_views, labels=labels)

            # Ensure dimensions are divisible by 2 for video encoding (libx264 requirement)
            h, w = combined.shape[:2]
            new_w = w if w % 2 == 0 else w + 1
            new_h = h if h % 2 == 0 else h + 1
            if new_w != w or new_h != h:
                padded = np.ones((new_h, new_w, 4), dtype=np.uint8) * 255
                padded[:, :, 3] = 255
                padded[:h, :w] = combined
                combined = padded

            # Convert RGBA to RGB with white background for saving
            combined_rgb = Image.new('RGB', (combined.shape[1], combined.shape[0]), (255, 255, 255))
            combined_rgba = Image.fromarray(combined, mode='RGBA')
            combined_rgb.paste(combined_rgba, mask=combined_rgba.split()[3])

            # Add text annotation at the bottom
            if text:
                combined_rgb = self.add_text(combined_rgb, text)

            seq_render_path = makepath(os.path.join(out_dir, seq_name, str(frame) + '.png'), isfile=True)
            combined_rgb.save(seq_render_path)

        # Restore original background color
        self.scene.bg_color = original_bg

        if T > 1:
            self.save_as_mp4(render_path=os.path.join(out_dir, seq_name), fps=fps)

        return os.path.join(out_dir, seq_name + ".mp4")

    
    def visualize_hands(self, out_dir, seq_name, pred, faces, skip_frame=1, fps=8, text=None):

        if text is None:
            text = seq_name
    
        T = pred.shape[0]
        for frame in tqdm.tqdm(np.arange(0,T, skip_frame), desc="rendering frames"):
            all_mesh = []
            # if frame == 0:
            #     self._center_cam(all_mesh)

            all_mesh.append(Mesh(vertices=pred[frame], faces=faces, vc=colors["yellow"], smooth=True))
            self.set_static_meshes(all_mesh)

            seq_render_path = makepath(os.path.join(out_dir, seq_name, str(frame) + '.png'),  isfile=True)
            self.save_snapshot(seq_render_path, text=text)

        if T > 1:
            self.save_as_mp4(render_path=os.path.join(out_dir, seq_name), fps=fps)

        return os.path.join(out_dir, seq_name+".mp4")
    
    
    def visualize_hands_with_gt_as_diff_views(self, out_dir, seq_name, pred, gt, faces, skip_frame=1, fps=8, text=None):
    
        
        if text is None:
            text = seq_name

        T = pred.shape[0]
        for frame in tqdm.tqdm(np.arange(0,T, skip_frame), desc="rendering frames"):
            all_mesh = [(Mesh(vertices=gt[frame], faces=faces, vc=colors["pink"], smooth=True))]
            self.set_static_meshes(all_mesh)
            gt_img, _ = self.viewer.render(self.scene)
    
            all_mesh = [(Mesh(vertices=pred[frame], faces=faces, vc=colors["yellow"], smooth=True))]
            self.set_static_meshes(all_mesh)
            pred_img, _ = self.viewer.render(self.scene)

            # hstack the image gt and pred
            out_img = np.concatenate([gt_img, pred_img], axis=1)
            seq_render_path = makepath(os.path.join(out_dir, seq_name, str(frame) + '.png'),  isfile=True)
            self.dump_snapshot(out_img, seq_render_path, text=text)

        if T > 1:
            self.save_as_mp4(render_path=os.path.join(out_dir, seq_name), fps=fps)

        return os.path.join(out_dir, seq_name+".mp4")
    

    def visualize_hands_with_gt_as_diff_views_with_obj(self, out_dir, seq_name, pred_hands_and_obj, gt_hands_and_obj, faces_hands_and_obj, skip_frame=1, fps=8, text=None):
      
        if text is None:
            text = seq_name

        gt_hands, gt_obj = gt_hands_and_obj
        pred_hands, pred_obj = pred_hands_and_obj
        faces_hands, faces_obj = faces_hands_and_obj
        T = pred_hands.shape[0]

        for frame in tqdm.tqdm(np.arange(0,T, skip_frame), desc="rendering frames"):

            ## generate GT
            all_mesh = [(Mesh(vertices=gt_hands[frame], faces=faces_hands, vc=colors["pink"], smooth=True))]
            all_mesh.append(Mesh(vertices=gt_obj[frame], faces=faces_obj, vc=colors["red"], smooth=True))

            self.set_static_meshes(all_mesh)
            gt_img, _ = self.viewer.render(self.scene)
            # gt_img = self.add_text(gt_img, "GT", (20, 20), (255, 0, 0))
            # cv2.putText(gt_img, "GT", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
            all_mesh = [(Mesh(vertices=pred_hands[frame], faces=faces_hands, vc=colors["yellow"], smooth=True))]
            all_mesh.append(Mesh(vertices=pred_obj[frame], faces=faces_obj, vc=colors["red"], smooth=True))
            self.set_static_meshes(all_mesh)
            pred_img, _ = self.viewer.render(self.scene)
            # cv2.putText(pred_img, "Pred", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            

            # hstack the image gt and pred
            out_img = np.concatenate([gt_img, pred_img], axis=1)
            seq_render_path = makepath(os.path.join(out_dir, seq_name, str(frame) + '.png'),  isfile=True)
            self.dump_snapshot(out_img, seq_render_path, text=text)

        if T > 1:
            self.save_as_mp4(render_path=os.path.join(out_dir, seq_name), fps=fps)

        return os.path.join(out_dir, seq_name+".mp4")


def points2sphere(points, radius = .001, vc = [0., 0., 1.], count = [5,5]):

    points = points.reshape(-1,3)
    n_points = points.shape[0]

    spheres = []
    for p in range(n_points):
        sphs = trimesh.creation.uv_sphere(radius=radius, count = count)
        sphs.apply_translation(points[p])
        sphs = Mesh(vertices=sphs.vertices, faces=sphs.faces, vc=vc)

        spheres.append(sphs)

    spheres = Mesh.concatenate_meshes(spheres)
    return spheres


def makepath(desired_path, isfile = False):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    import os
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path

colors = {
    'pink': [1.00, 0.75, 0.80],
    'purple': [0.63, 0.13, 0.94],
    'red': [1.0, 0.0, 0.0],
    'green': [.0, 1., .0],
    'yellow': [1., 1., 0],
    'brown': [1.00, 0.25, 0.25],
    'blue': [.0, .0, 1.],
    'white': [1., 1., 1.],
    'orange': [1.00, 0.65, 0.00],
    'grey': [0.75, 0.75, 0.75],
    'black': [0., 0., 0.],
}


def create_cylinder(length=1.0, radius=0.05, axis='z'):
    """
    Creates a cylinder aligned along the specified axis, extending only in the positive direction.

    Args:
        length (float): Length of the cylinder.
        radius (float): Radius of the cylinder.
        axis (str): Which axis ('x', 'y', or 'z') the cylinder should align to.

    Returns:
        trimesh.Trimesh: Cylinder mesh transformed to align with the specified axis.
    """
    # Create a default cylinder along Z-axis
    cylinder = trimesh.creation.cylinder(radius=radius, height=length)

    # Move the cylinder so it starts at (0,0,0) and extends in the positive direction
    transform = np.eye(4)  # Identity matrix

    if axis == 'x':
        cylinder.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))  
        transform[0, 3] = length / 2  # Move along positive X
    elif axis == 'y':
        cylinder.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))  
        transform[1, 3] = length / 2  # Move along positive Y
    elif axis == 'z':
        transform[2, 3] = length / 2  # Move along positive Z

    cylinder.apply_transform(transform)
    return cylinder

def create_sphere(radius=0.1, color=[255, 0, 0, 255]):
    """
    Creates a sphere at the origin (0,0,0) to mark the origin.

    Args:
        radius (float): Radius of the sphere.
        color (list): RGBA color of the sphere.

    Returns:
        trimesh.Trimesh: Sphere mesh.
    """
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)  # Create a smooth sphere
    sphere.visual.vertex_colors = color  # Set the sphere color (Red)
    return sphere

def origin_marker_geometry(axis_length=1.0, radius=0.05, sphere_radius=0.1):
    """
    Generates a 3D origin marker with X, Y, and Z cylinders, extending only in the positive direction.
    Also adds a red sphere at (0,0,0) to mark the origin.

    Args:
        axis_length (float): Length of each axis.
        radius (float): Radius of each cylinder.
        sphere_radius (float): Radius of the origin sphere.

    Returns:
        trimesh.Trimesh: Combined mesh of the origin marker.
    """
    # Create positive-only cylinders for each axis
    x_axis = create_cylinder(length=axis_length, radius=radius, axis='x')
    y_axis = create_cylinder(length=axis_length, radius=radius, axis='y')
    z_axis = create_cylinder(length=axis_length, radius=radius, axis='z')

    # Create a red sphere at the origin
    # origin_sphere = create_sphere(radius=sphere_radius, color=[255, 0, 0, 255])

    # Assign colors: Red (X), Green (Y), Blue (Z)
    x_axis.visual.vertex_colors = [255, 0, 0, 255]  # Red for X-axis
    y_axis.visual.vertex_colors = [0, 255, 0, 255]  # Green for Y-axis
    z_axis.visual.vertex_colors = [0, 0, 255, 255]  # Blue for Z-axis

    # Combine all objects into a single mesh
    origin_marker = trimesh.util.concatenate([x_axis, y_axis, z_axis])

    return origin_marker


#### cube creation
def create_cylinder_v2(length=1.0, radius=0.05, axis='z'):
    """
    Creates a cylinder aligned along the specified axis, extending symmetrically.

    Args:
        length (float): Length of the cylinder.
        radius (float): Radius of the cylinder.
        axis (str): Which axis ('x', 'y', or 'z') the cylinder should align to.

    Returns:
        trimesh.Trimesh: Cylinder mesh transformed to align with the specified axis.
    """
    # Create a default cylinder along the Z-axis
    cylinder = trimesh.creation.cylinder(radius=radius, height=length)

    # Transform based on the axis
    if axis == 'x':
        cylinder.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    elif axis == 'y':
        cylinder.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))

    return cylinder

def create_cube_edges(size=1.0, radius=0.05):
    """
    Creates a cube with cylindrical edges, centered at the origin.
    Each edge cylinder is colored based on its parallel axis.

    Args:
        size (float): Length of the cube's edges.
        radius (float): Radius of the cylinders forming the edges.

    Returns:
        trimesh.Trimesh: Mesh containing the cube edges as cylinders.
    """
    offsets = [-size / 2, size / 2]
    edges = []

    # Define colors for different axis-aligned edges
    axis_colors = {'x': [255, 0, 0, 255],   # Red for X-aligned edges
                   'y': [0, 255, 0, 255],   # Green for Y-aligned edges
                   'z': [0, 0, 255, 255]}  # Blue for Z-aligned edges

    # Generate edges along each axis
    for axis in ['x', 'y', 'z']:
        for a in offsets:
            for b in offsets:
                # Create edge cylinder
                edge = create_cylinder_v2(length=size, radius=radius, axis=axis)

                # Position the cylinder correctly
                if axis == 'x':
                    edge.apply_translation([0, a, b])  # X-aligned edges
                elif axis == 'y':
                    edge.apply_translation([a, 0, b])  # Y-aligned edges
                elif axis == 'z':
                    edge.apply_translation([a, b, 0])  # Z-aligned edges

                # Assign the corresponding color
                edge.visual.vertex_colors = axis_colors[axis]
                edges.append(edge)

    # Combine all edges into a single mesh
    cube_edges = trimesh.util.concatenate(edges)

    return cube_edges

#### create cylinder to position

def create_cylinder_to_position(position, radius=0.007):
    """
    Creates a cylinder from the origin (0,0,0) to a given 3D position.

    Args:
        position (np.array): Target position as a (3,) numpy array [x, y, z].
        radius (float): Radius of the cylinder.

    Returns:
        trimesh.Trimesh: Cylinder from the origin to the given position.
    """
    position = np.array(position)  # Ensure it's a numpy array
    position = position * 1.5
    print("Enabled scaling for debugging")
    # Compute the length (distance from origin)
    length = np.linalg.norm(position)

    if length == 0:
        raise ValueError("Position cannot be the origin itself.")

    # Create a default cylinder along the Z-axis
    cylinder = trimesh.creation.cylinder(radius=radius, height=length)

    # Compute the rotation required to align cylinder from Z-axis to target direction
    z_axis = np.array([0, 0, 1])  # Default cylinder axis
    direction = position / length  # Normalize the direction vector
    rotation_matrix = trimesh.geometry.align_vectors(z_axis, direction)  # Compute rotation

    # Apply rotation & translation
    cylinder.apply_transform(rotation_matrix)  # Rotate to align with direction
    cylinder.apply_translation(position / 2)  # Move to midpoint

    return cylinder

def camera_marker_geometry(radius, height):
    vertices = np.array(
        [
            [-radius, -radius, 0],
            [radius, -radius, 0],
            [radius, radius, 0],
            [-radius, radius, 0],
            [0, 0, - height],
        ]
    )


    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [1, 0, 4], [2, 1, 4], [3, 2, 4], [0, 3, 4],]
    )

    face_colors = np.array(
        [
            [0.5, 0.5, 0.5, 1.0],
            [0.5, 0.5, 0.5, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ]
    )
    return vertices, faces, face_colors
