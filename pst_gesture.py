import numpy as np

ADAM_U_NUM_MOTOR = 19
KP_CONFIG = [
    60.0,  # waistRoll (0)
    60.0,  # waistPitch (1)
    60.0,  # waistYaw (2)
    9.0,  # neckYaw (3)
    9.0,  # neckPitch (4)
    18.0,  # shoulderPitch_Left (5)
    9.0,  # shoulderRoll_Left (6)
    9.0,  # shoulderYaw_Left (7)
    9.0,  # elbow_Left (8)
    9.0,  # wristYaw_Left (9)
    9.0,  # wristPitch_Left (10)
    9.0,  # wristRoll_Left (11)
    18.0,  # shoulderPitch_Right (12)
    9.0,  # shoulderRoll_Right (13)
    9.0,  # shoulderYaw_Right (14)
    9.0,  # elbow_Right (15)
    9.0,  # wristYaw_Right (16)
    9.0,  # wristPitch_Right (17)
    9.0,  # wristRoll_Right (18)
]

# Kd 配置数组（对应19个关节）
KD_CONFIG = [
    1.0,  # waistRoll (0)
    1.0,  # waistPitch (1)
    1.0,  # waistYaw (2)
    0.9,  # neckYaw (3)
    0.9,  # neckPitch (4)
    0.9,  # shoulderPitch_Left (5)
    0.9,  # shoulderRoll_Left (6)
    0.9,  # shoulderYaw_Left (7)
    0.9,  # elbow_Left (8)
    0.9,  # wristYaw_Left (9)
    0.9,  # wristPitch_Left (10)
    0.9,  # wristRoll_Left (11)
    0.9,  # shoulderPitch_Right (12)
    0.9,  # shoulderRoll_Right (13)
    0.9,  # shoulderYaw_Right (14)
    0.9,  # elbow_Right (15)
    0.9,  # wristYaw_Right (16)
    0.9,  # wristPitch_Right (17)
    0.9,  # wristRoll_Right (18)
]

HOME = np.array(
    [
        0.0,  # 0: waistRoll
        0.0,  # 1: waistPitch
        0.0,  # 2: waistYaw
        0.0,  # 3: neckYaw
        0.0,  # 4: neckPitch
        0.0,  # 5: shoulderPitch_Left
        0.0,  # 6: shoulderRoll_Left
        0.0,  # 7: shoulderYaw_Left
        0.0,  # 8: elbow_Left
        0.0,  # 9: wristYaw_Left
        0.0,  # 10: wristPitch_Left
        0.0,  # 11: wristRoll_Left
        0.0,  # 12: shoulderPitch_Right
        0.0,  # 13: shoulderRoll_Right
        0.0,  # 14: shoulderYaw_Right
        0.0,  # 15: elbow_Right
        0.0,  # 16: wristYaw_Right
        0.0,  # 17: wristPitch_Right
        0.0,  # 18: wristRoll_Right
    ]
)

READY = np.array(
    [
        0.0,  # 0: waistRoll
        0.0,  # 1: waistPitch
        0.0,  # 2: waistYaw
        0.0,  # 3: neckYaw
        0.0,  # 4: neckPitch
        0.0,  # 5: shoulderPitch_Left
        0.0,  # 6: shoulderRoll_Left
        0.0,  # 7: shoulderYaw_Left
        0.0,  # 8: elbow_Left
        0.0,  # 9: wristYaw_Left
        0.0,  # 10: wristPitch_Left
        0.0,  # 11: wristRoll_Left
        -0.2618,  # 12: shoulderPitch_Right
        0.0,  # 13: shoulderRoll_Right
        0.5236,  # 14: shoulderYaw_Right
        -2.0944,  # 15: elbow_Right
        -1.0472,  # 16: wristYaw_Right
        0.0,  # 17: wristPitch_Right
        0.0,  # 18: wristRoll_Right
    ]
)

EXE = np.array(
    [
        0.0,  # 0: waistRoll
        0.0,  # 1: waistPitch
        0.0,  # 2: waistYaw
        0.0,  # 3: neckYaw
        0.0,  # 4: neckPitch
        0.0,  # 5: shoulderPitch_Left
        0.0,  # 6: shoulderRoll_Left
        0.0,  # 7: shoulderYaw_Left
        0.0,  # 8: elbow_Left
        0.0,  # 9: wristYaw_Left
        0.0,  # 10: wristPitch_Left
        0.0,  # 11: wristRoll_Left
        -0.2618,  # 12: shoulderPitch_Right
        0.0,  # 13: shoulderRoll_Right
        0.5236,  # 14: shoulderYaw_Right
        -1.5708,  # 15: elbow_Right
        -1.0472,  # 16: wristYaw_Right
        0.0,  # 17: wristPitch_Right
        0.0,  # 18: wristRoll_Right
    ],
    dtype=float,
)

ROCK = np.array(
    [
        0,  # 0: L_pinky
        0,  # 1: L_ring
        0,  # 2: L_middle
        0,  # 3: L_index
        200,  # 4: L_thumb_2
        1000,  # 5: L_thumb_1
        0,  # 6: R_pinky
        0,  # 7: R_ring
        0,  # 8: R_middle
        0,  # 9: R_index
        200,  # 10: R_thumb_2
        1000,  # 11: R_thumb_1
    ],
    dtype=int,
)

PAPER = np.array(
    [
        0,  # 0: L_pinky
        0,  # 1: L_ring
        0,  # 2: L_middle
        0,  # 3: L_index
        200,  # 4: L_thumb_2
        1000,  # 5: L_thumb_1
        1000,  # 6: R_pinky
        1000,  # 7: R_ring
        1000,  # 8: R_middle
        1000,  # 9: R_index
        1000,  # 10: R_thumb_2
        1000,  # 11: R_thumb_1
    ],
    dtype=int,
)

SCISSORS = np.array(
    [
        0,  # 0: L_pinky
        0,  # 1: L_ring
        0,  # 2: L_middle
        0,  # 3: L_index
        200,  # 4: L_thumb_2
        1000,  # 5: L_thumb_1
        0,  # 6: R_pinky
        0,  # 7: R_ring
        1000,  # 8: R_middle
        1000,  # 9: R_index
        200,  # 10: R_thumb_2
        1000,  # 11: R_thumb_1
    ],
    dtype=int,
)
