from stardist.data import test_image_nuclei_3d 
from stardist.rays3d import Rays_GoldenSpiral
from stardist.geometry import star_dist3D, starflow3d

if __name__ == "__main__":
    img, mask = test_image_nuclei_3d(return_mask=True)
    
    rays = Rays_GoldenSpiral(32)
    
    grid = (1,1,1)
    dist = star_dist3D(mask, rays, grid)
    
    
    flow = starflow3d(dist, mask, rays, grid)
    
    