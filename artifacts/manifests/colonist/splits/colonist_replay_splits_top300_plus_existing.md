# Colonist Replay Splits

Generated: 2026-05-15T04:08:56.481164+00:00
Total games: 7424

## Split Counts

| Split | Games | Raw replays present |
| --- | ---: | ---: |
| train | 5936 | 0 |
| val | 738 | 0 |
| test | 750 | 0 |

## Balance Color Counts

### train

| Color | Games |
| --- | ---: |
| BLACK | 1473 |
| BLUE | 1272 |
| BRONZE | 49 |
| GOLD | 55 |
| GREEN | 636 |
| MYSTIC_BLUE | 71 |
| ORANGE | 1070 |
| RED | 1310 |

### val

| Color | Games |
| --- | ---: |
| BLACK | 184 |
| BLUE | 159 |
| BRONZE | 6 |
| GOLD | 6 |
| GREEN | 79 |
| MYSTIC_BLUE | 8 |
| ORANGE | 133 |
| RED | 163 |

### test

| Color | Games |
| --- | ---: |
| BLACK | 185 |
| BLUE | 159 |
| BRONZE | 7 |
| GOLD | 8 |
| GREEN | 80 |
| MYSTIC_BLUE | 10 |
| ORANGE | 135 |
| PINK | 1 |
| RED | 165 |

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
