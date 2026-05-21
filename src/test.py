import cv2
import py360convert

# Load your 2D flat equirectangular image
equi_img = cv2.imread('frame_000690.jpg')
if equi_img is None:
    raise FileNotFoundError("Could not read frame_000690.jpg")

# Extract a normal perspective shot from it
# h_fov and v_fov set the field of view; u_deg and v_deg set the viewing angles
perspective_img = py360convert.e2p(
    equi_img,
    fov_deg=(90, 60),
    u_deg=0,    # Yaw (horizontal look angle)
    v_deg=0,    # Pitch (vertical look angle)
    out_hw=(600, 800)  # Output image dimensions
)

cv2.imwrite('output_perspective.jpg', perspective_img)
