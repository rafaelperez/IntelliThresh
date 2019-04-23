# tags:
# hole, occlusion, gray, color, upper, lower, confusion, pose, front, back, side
# source: ChengleiSocialFull
test_set_tag = 'dani'
data_root = '/media/mscv1/14fecf86-bdfa-4ebd-8b47-eea4ddee198e/dome_fg_imgs/newfgimg'
bg_dir = '000000'

test_img_list = [
    ('030000', '400128', {'color'}),
    ('030000', '410133', {'gray'}),
    ('030240', '400145', {'color', 'hole'}),
    ('030240', '400239', {'color', 'hole'}),
    ('030240', '410149', {'gray', 'hole'}),
    ('030240', '410223', {'gray', 'hole'}),
    ('030240', '410168', {'gray', 'hole'}),
    ('030240', '410263', {'gray', 'lower'}),
]