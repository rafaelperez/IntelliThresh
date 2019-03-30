# tags:
# hole, occlusion, gray, color, upper, lower, confusion, pose, front, back, side
# source: ChengleiSocialFull
test_set_tag = 'chenglei_social_full'
data_root = '/home/mscv1/Desktop/FRL/ChengleiSocial_full/ChengleiSocial/undistorted'
bg_dir = '000000'

test_img_list = [
    ('020321', '400080', {'hole', 'color', 'front'}),
    ('020321', '400082', {'color', 'side'}),
    ('020321', '400146', {'color', 'back', 'confusion'}),
    ('020321', '400164', {'color', 'back', 'confusion', 'hole'}),
    ('020321', '400233', {'hole', 'color', 'back'}),
    ('020321', '400200', {'color', 'side', 'upper'}),
    ('020321', '400183', {'color', 'front', 'occlusion'}),
    ('020321', '400185', {'color', 'back', 'upper', 'confusion'}),
    
    ('020321', '410086', {'gray', 'front', 'upper'}),
    ('020321', '410149', {'gray', 'back', 'upper'}),
    ('020321', '410134', {'gray', 'front', 'lower'}),
    ('020321', '410144', {'gray', 'side', 'lower'}),
    ('020321', '410163', {'gray', 'back', 'lower'}),

    ('020979', '400058', {'color', 'hole', 'pose', 'side'}),
    ('020979', '400075', {'color', 'hole', 'pose', 'front'}),
    ('020979', '400133', {'color', 'hole', 'pose', 'back'}),
    ('020979', '400181', {'color', 'hole', 'pose', 'side'}),
    ('020979', '400179', {'color', 'confusion', 'pose', 'back'}),
    ('020979', '400183', {'color', 'front', 'occlusion', 'pose'}),

    ('020979', '410194', {'gray', 'hole', 'pose', 'side'}),
    ('020979', '410202', {'gray', 'pose', 'front'}),
    ('020979', '410247', {'gray', 'pose', 'back', 'confusion'}),
    ('020979', '410254', {'gray', 'pose', 'front'})   
]