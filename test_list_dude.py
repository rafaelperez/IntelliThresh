# tags:
# hole, occlusion, gray, color, upper, lower, confusion, pose, front, back, side
# source: ChengleiSocialFull
test_set_tag = 'dude'
data_root = '/media/mscv1/14fecf86-bdfa-4ebd-8b47-eea4ddee198e/dome_fg_imgs/dude'
bg_dir = '000000'

test_img_list = [
    ('000001', '400054', {'color', 'confuse'}),
    ('000001', '400160', {'color', 'birdview'}),
    ('000001', '400204', {'color', 'hole'}),
    ('000001', '410095', {'gray', 'birdview'}),
    ('000001', '400167', {'color', 'confuse'}),
    ('000001', '400210', {'color', 'confuse'}),
    ('000001', '410263', {'gray', 'confuse', 'lower'})
]