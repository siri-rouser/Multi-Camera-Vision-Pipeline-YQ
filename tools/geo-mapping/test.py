import cameratransform as ct

camSB = ct.load_camera('fitted_camSB.json')

loc = camSB.gpsFromImage([985,308],Z=0)

print(loc)