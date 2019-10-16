import bpy
import pickle

        
all_primitives = pickle.load(open('/home/dima/Documents/research/DeepParametricShapes/boxes', 'rb'))

for name, primitives in all_primitives.items():
    objects = bpy.data.objects
    for obj in objects:
        if obj.type == 'MESH':
            obj.select_set(True)
            bpy.ops.object.delete()
            
    boxes = []
    for box in primitives:
        bpy.ops.mesh.primitive_cube_add(location=box['translation'])
        bpy.ops.transform.resize(value=box['size'])
        box_ = bpy.context.active_object
        box_.rotation_mode = 'QUATERNION'
        box_.data.materials.append(bpy.data.materials['orange'])
        rot = box['rotation']
        box_.rotation_quaternion = (-rot[0], rot[1], rot[2], rot[3])
        if 'r' in box:
            bevel = box_.modifiers.new('bevel', 'BEVEL')
            bevel.segments = 40
            bevel.width = box['r']
        boxes.append(box_)
    
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = f'/home/dima/Documents/research/renders/{name}.png'
    bpy.ops.render.render(write_still=1)

# spheres = []
# for sphere in primitives['spheres']:
#     bpy.ops.mesh.primitive_cube_add(location=sphere['translation'])
#     bpy.ops.transform.resize(value=sphere['size'])
#     sphere_ = bpy.context.active_object
#     sphere_.rotation_mode = 'QUATERNION'
#     rot = sphere['rotation']
#     sphere_.rotation_quaternion = (-rot[0], rot[1], rot[2], rot[3])
#     spheres.append(sphere_)

#     for box in boxes:
#         bool = box.modifiers.new('bool', 'BOOLEAN')
#         bool.operation = 'DIFFERENCE'
#         bool.object = sphere_
#     sphere_.hide = True
