# Gravity Rubber Simulator

Computer Graphics homework, programmed in C++ with OpenGL.

## Task specification

Gravity rubber simulator. Our rubber sheet with a flat torus topology is initially viewed from above, on which we can place large, non-moving bodies by pressing the right mouse button. By pressing the left mouse button, you can slide lightweight balls from the lower left corner without friction. The location of the click, along with the lower left corner, specifies the starting speed and direction. Resting heavy bodies curve the space, i.e. deform the rubber sheet, but they are not visible. The indentation caused at a distance r from the center of the mass is m / (r + r0), where r0 is half the width of the rubber sheet and m is an increasing mass on successive bodies. The rubber sheet is optically shabby, with a diffuse and ambient factor that darkens gradually according to the depth. The balls are color-diffuse-specular, with negligible spatial curvature and size. When you press SPACE, our virtual camera sticks to the first bullet not yet absorbed, so we can follow its point of view as well. The balls that hit the masses are absorbed, the collision between the balls does not have to be dealt with. The rubber sheet is illuminated by two point light sources which rotate around each other's initial position according to the following quaternion (t is the time): q = [cos (t / 4), sin (t / 4) cos (t) / 2, sin (t / 4) sin (t) / 2, sin (t / 4) √ (3/4])

## Solution

YouTube video showcasing the solution: https://youtu.be/qNmL1yJAUAk

<img width="595" alt="Screenshot 2022-03-27 at 22 23 08" src="https://user-images.githubusercontent.com/27449756/160299771-79cc36d4-ea45-4b3c-855d-d4192ff2883e.png">
<img width="595" alt="Screenshot 2022-03-27 at 22 23 26" src="https://user-images.githubusercontent.com/27449756/160299774-cb64e70f-c47d-4f78-9816-354dc6ae5616.png">
<img width="595" alt="Screenshot 2022-03-27 at 22 23 48" src="https://user-images.githubusercontent.com/27449756/160299779-7e8429ba-5a3b-4e7d-92c9-1d9cb2b2ad57.png">

