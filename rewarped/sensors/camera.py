import numpy as np
import torch
import warp as wp

class BaseSensor:
    """Base class for all sensors."""
    
    def __init__(self, uid):
        self.uid = uid
        
    def setup(self):
        """Setup this sensor. Called during environment initialization."""
        pass
        
    def capture(self):
        """
        Captures sensor data and prepares it for retrieval.
        This should be called after environment.update_render().
        """
        pass
        
    def get_obs(self):
        """
        Retrieves captured sensor data as an observation for the agent.
        """
        raise NotImplementedError()
    
    
class Camera(BaseSensor):
    """A camera sensor for WarpEnv."""
    
    def __init__(self, uid, position, target, up=[0, 0, 1], width=84, height=84, fov=60.0):
        super().__init__(uid)
        self.position = position
        self.target = target
        self.up = up
        self.width = width
        self.height = height
        self.fov = fov
        self._data = None
        
    def setup(self):
        """Initialize the camera for rendering if needed."""
        # In a complete implementation, this would set up any necessary rendering components
        pass
        
    def capture(self, model):
        """
        Captures image data from the current model state.
        
        Args:
            model: The Warp model to render
            
        Returns:
            None (data is stored internally)
        """
        
        # Get renderer from model
        renderer = model.renderer
        if renderer is None:
            return None
            
        # Set camera parameters
        renderer.camera_position = self.position 
        renderer.camera_target = self.target
        renderer.camera_up = self.up
        renderer.camera_fov = self.fov
        renderer.render_width = self.width
        renderer.render_height = self.height
        
        # Render the scene
        renderer.begin_frame()
        renderer.render(model)
        renderer.end_frame()
        
        # Get rendered data
        rgb = renderer.get_rgb_image()
        depth = renderer.get_depth_image() 
        segmentation = renderer.get_segmentation_image()
        # In a real implementation, this would use Warp's rendering capabilities
        # For now, we'll create dummy data
        rgb = torch.zeros((self.height, self.width, 3), dtype=torch.uint8)
        depth = torch.zeros((self.height, self.width), dtype=torch.float32)
        segmentation = torch.zeros((self.height, self.width), dtype=torch.int32)
        
        self._data = {
            "rgb": rgb,
            "depth": depth,
            "segmentation": segmentation
        }
        
        return self._data
        
    def get_obs(self, rgb=True, depth=False, segmentation=False):
        """
        Returns the captured sensor data as observation.
        
        Args:
            rgb: Whether to include RGB data
            depth: Whether to include depth data
            segmentation: Whether to include segmentation data
            
        Returns:
            Dict containing the requested observation data
        """
        if self._data is None:
            return None
            
        obs = {}
        if rgb and "rgb" in self._data:
            obs["rgb"] = self._data["rgb"]
        if depth and "depth" in self._data:
            obs["depth"] = self._data["depth"]
        if segmentation and "segmentation" in self._data:
            obs["segmentation"] = self._data["segmentation"]
            
        return obs if len(obs) > 1 else next(iter(obs.values()))