import pandas as pd
import re
import numpy as np

GRADES = ['6A+', '6B', '6B+', '6C', '6C+', '7A', '7A+', '7B', '7B+', '7C', '7C+', '8A', '8A+', '8B', '8B+']
GRADE_DICT = {grade: i for i, grade in enumerate(GRADES)}
GRADE_DICT_REVERSE = {i: grade for i, grade in enumerate(GRADES)}

def get_df(path: str) -> pd.DataFrame:
    with open(path, "r") as file:
        data = pd.read_json(file)

    # Normalize the data
    main_df = pd.json_normalize(data['data'])
    moves_df = pd.json_normalize(data=data['data'], record_path='moves')

    # create dummy variables for the hold, i.e. description column
    move_dummies = pd.get_dummies(moves_df.description)
    moves_df = pd.concat([moves_df, move_dummies], axis=1).drop_duplicates()

    df = main_df.merge(moves_df, left_on='apiId', right_on='problemId')
    
    df = df.drop(columns=['moves', 'hasBetaVideo', 'holdsets', 'setbyId', 'userGrade', 'moonBoardConfigurationId',
                      'holdsetup.description', 'holdsetup.apiId', 'holdsetup.holdsets', 'description', 'apiId',
                      'dateUpdated', 'isMaster'])
    
    return df

def prepare_features(df: pd.DataFrame, min_repeats=5, benchmark=False) -> pd.DataFrame:
    # filter out deleted problems
    df = df[df.dateDeleted.isnull()]
    if benchmark:
        df = df[df.isBenchmark == True]
    # filter out problems with no grade
    df = df.query("grade != 'None'")
    df = df.assign(
        # convert grades to numeric
        grade=df.grade.map(GRADE_DICT),
        # convert date to datetime & extract year
        year = pd.to_datetime(df.dateInserted, format='ISO8601').dt.year,
    )
    df = df.drop(columns=['dateInserted', 'dateDeleted'])
    # filter out problems with less than min_repeats repeats
    df = df.query(f"repeats >= {min_repeats}")
    
    return df

def group_by_problem(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Group by all the columns that are not holds (A1 - K9)
    2) For each group (problem), sum over the holds to get dummies for each boulder
    3) Mark the start hold and the end hold by as the hold per group where isStart / isEnd is true
    """
    df.set_index('problemId', inplace=True)
    # Identify columns matching the pattern: single letter followed by 1 or 2 digits
    hold_cols = [col for col in df.columns if re.match(r'^[A-Z]\d{1,2}$', col)]

    # All other columns are non-hold columns
    non_hold_cols = [col for col in df.columns if col not in hold_cols]

    # Group by non-hold columns and sum over the hold columns
    grouped = (df
    .groupby('problemId')
    [hold_cols]
    .sum())

    # Add the non-hold columns back to the grouped DataFrame
    grouped = (df[non_hold_cols]
    .drop(columns=['isStart', 'isEnd'])
    .drop_duplicates()).merge(grouped, on='problemId')

    # find the hold_col that is true
    start_holds = df.query('isStart == True')[hold_cols].idxmax(axis=1)
    end_holds = df.query('isEnd == True')[hold_cols].idxmax(axis=1)
    start_holds.name = 'startHold'
    end_holds.name = 'endHold'

    # There can be one or two start holds, so we need to group them
    start_holds_grouped = start_holds.groupby(start_holds.index).apply(list)
    # remove boulders with more than 2 start holds
    start_holds_grouped = start_holds_grouped[start_holds_grouped.apply(len) <= 2]
    start_hold_df = pd.DataFrame(start_holds_grouped.tolist(), index=start_holds_grouped.index)
    start_hold_df.columns = ['start_hold_1', 'start_hold_2']
    start_hold_df['start_hold_2'] = start_hold_df['start_hold_2'].where(pd.notna(start_hold_df['start_hold_2']), np.nan)

    # Merge the start and end holds with the grouped DataFrame
    grouped = grouped.merge(start_hold_df, left_index=True, right_index=True).merge(end_holds, left_index=True, right_index=True)
    return grouped


