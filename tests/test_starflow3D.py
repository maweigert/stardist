from stardist.data import test_image_nuclei_3d 
from stardist.rays3d import Rays_GoldenSpiral
from stardist.geometry import star_dist3D, starflow3d, starflow3d_map
from stardist.utils import _edt_prob_edt
from stardist.models import StarDist3D
import numpy as np
from scipy.ndimage import zoom 


if __name__ == "__main__":
    img, mask = test_image_nuclei_3d(return_mask=True)
    
    # model = StarDist3D.from_pretrained('3D_demo')
    
    # rays = Rays_GoldenSpiral(32)
    
    # grid = (2,2,2)
    # dist = star_dist3D(mask, rays, grid)
    
    # flow = starflow3d(dist, mask, rays, grid)
    
    # sc = tuple(s1/s2 for s1, s2 in zip(img.shape, dist.shape))
        
    # mask_shrunk = (_edt_prob_edt(mask, anisotropy=(5,1,1))>0.5)*mask 
    # fg = mask>0
    # flow_scaled = zoom(flow,sc+(1,), order=1)
    
    # #final = starflow3d_map(flow_scaled, mask, fg_scaled, delta=0.5)
    
    # final = tuple(starflow3d_map(flow_scaled, mask_shrunk, fg, delta=0.1, rounds=r) for r in range(1,100,5))
    # final = np.stack(final, axis=0)
    
    
    # print(mask.shape)
    # print(flow.shape)
    # print(np.max(flow))
    
    # flow_abs = np.linalg.norm(flow_scaled, axis=-1)
    # import napari 
    
    # v = napari.Viewer()
    # # v.add_image(flow_abs, colormap='magma')
    # v.add_labels(final)
    # # # v.add_image(dist, colormap='magma')
    # # v.add_image(np.moveaxis(flow, -1, 0), colormap='magma')
    
    # # napari.run()