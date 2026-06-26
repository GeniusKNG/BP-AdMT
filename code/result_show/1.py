import pyvista as pv
import numpy as np
from skimage import measure
import h5py

data = h5py.File(r'D:\pycode\BP-AdMT\data\Pancreas\Pancreas_h5\image0001_norm.h5', 'r')
prediction = data['image'][:]
label = data['label'][:]


# marching cubes
verts, faces, normals, values = measure.marching_cubes(label, level=0.5)

# pyvista mesh
faces = np.hstack(
    [np.full((faces.shape[0],1),3), faces]
).astype(np.int64)

mesh = pv.PolyData(verts, faces)

mesh = mesh.smooth(
    n_iter=600,
    relaxation_factor=0.01
)

# plot
plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(mesh, color='red')

# plotter.show()
plotter.set_background("white")
plotter.screenshot("1.png")
