# Colonist Replay Splits

Generated: 2026-05-15T03:58:45.050765+00:00
Total games: 7369

## Split Counts

| Split | Games | Raw replays present |
| --- | ---: | ---: |
| train | 5891 | 0 |
| val | 733 | 0 |
| test | 745 | 0 |

## Balance Color Counts

### train

| Color | Games |
| --- | ---: |
| BLACK | 1450 |
| BLUE | 1268 |
| BRONZE | 49 |
| GOLD | 55 |
| GREEN | 626 |
| MYSTIC_BLUE | 71 |
| ORANGE | 1068 |
| RED | 1304 |

### val

| Color | Games |
| --- | ---: |
| BLACK | 181 |
| BLUE | 158 |
| BRONZE | 6 |
| GOLD | 6 |
| GREEN | 78 |
| MYSTIC_BLUE | 8 |
| ORANGE | 133 |
| RED | 163 |

### test

| Color | Games |
| --- | ---: |
| BLACK | 182 |
| BLUE | 160 |
| BRONZE | 7 |
| GOLD | 8 |
| GREEN | 79 |
| MYSTIC_BLUE | 10 |
| ORANGE | 135 |
| PINK | 1 |
| RED | 163 |

## Step Plans

`early_mid` is the current training target. `late_hard` is kept separate for a harder later dataset.

```json
{
  "early_mid": {
    "early": [
      0.08,
      0.18,
      0.28
    ],
    "mid": [
      0.38,
      0.5
    ]
  },
  "late_hard": {
    "late_hard": [
      0.62,
      0.74,
      0.86
    ]
  }
}
```
