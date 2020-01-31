
def bin_LotArea(df):
    """
        mean      10516.828082
        std        9981.264932
        min        1300.000000
        25%        7553.500000
        50%        9478.500000
        75%       11601.500000
        max      215245.000000
    """

    df.loc[ df['LotArea'] <= 2000, 'LotArea'] = 0,
    df.loc[(df['LotArea'] > 2000)   & (df['LotArea'] <= 4000),   'LotArea'] = 1,
    df.loc[(df['LotArea'] > 4000)   & (df['LotArea'] <= 6000),   'LotArea'] = 2,
    df.loc[(df['LotArea'] > 6000)   & (df['LotArea'] <= 8000),   'LotArea'] = 3,
    df.loc[(df['LotArea'] > 8000)   & (df['LotArea'] <= 10000),  'LotArea'] = 4,
    df.loc[(df['LotArea'] > 10000)  & (df['LotArea'] <= 25000),  'LotArea'] = 5,
    df.loc[(df['LotArea'] > 25000)  & (df['LotArea'] <= 50000),  'LotArea'] = 6,
    df.loc[(df['LotArea'] > 50000)  & (df['LotArea'] <= 100000), 'LotArea'] = 7,
    df.loc[ df['LotArea'] > 100000, 'LotArea'] = 8

    return df

def bin_YearBuilt(df):
    """
        min      1872.000000
        25%      1954.000000
        50%      1973.000000
        75%      2000.000000
        max      2010.000000
    """
        
    df.loc[ df['YearBuilt'] <= 1900, 'YearBuilt'] = 0,
    df.loc[(df['YearBuilt'] > 1900)   & (df['YearBuilt'] <= 1910),   'YearBuilt'] = 1,
    df.loc[(df['YearBuilt'] > 1910)   & (df['YearBuilt'] <= 1920),   'YearBuilt'] = 2,
    df.loc[(df['YearBuilt'] > 1920)   & (df['YearBuilt'] <= 1930),   'YearBuilt'] = 3,
    df.loc[(df['YearBuilt'] > 1930)   & (df['YearBuilt'] <= 1940),  'YearBuilt'] = 4,
    df.loc[(df['YearBuilt'] > 1940)  & (df['YearBuilt'] <= 1950),  'YearBuilt'] = 5,
    df.loc[(df['YearBuilt'] > 1950)  & (df['YearBuilt'] <= 1960),  'YearBuilt'] = 6,
    df.loc[(df['YearBuilt'] > 1960)  & (df['YearBuilt'] <= 1970), 'YearBuilt'] = 7,
    df.loc[(df['YearBuilt'] > 1970)  & (df['YearBuilt'] <= 1980), 'YearBuilt'] = 8,
    df.loc[(df['YearBuilt'] > 1980)  & (df['YearBuilt'] <= 1990), 'YearBuilt'] = 9,
    df.loc[(df['YearBuilt'] > 1990)  & (df['YearBuilt'] <= 2000), 'YearBuilt'] = 10,
    df.loc[ df['YearBuilt'] > 2000, 'YearBuilt'] = 11

    return df

def bin_GarageType(df):
    """
        count    1460.000000
        mean      472.980137
        std       213.804841
        min         0.000000
        25%       334.500000
        50%       480.000000
        75%       576.000000
        max      1418.000000
    """

    df.loc[ df['GarageArea'] <= 100, 'GarageArea'] = 0,
    df.loc[(df['GarageArea'] > 100)   & (df['GarageArea'] <= 200),   'GarageArea'] = 1,
    df.loc[(df['GarageArea'] > 200)   & (df['GarageArea'] <= 300),   'GarageArea'] = 2,
    df.loc[(df['GarageArea'] > 300)   & (df['GarageArea'] <= 400),   'GarageArea'] = 3,
    df.loc[(df['GarageArea'] > 400)   & (df['GarageArea'] <= 500),  'GarageArea'] = 4,
    df.loc[(df['GarageArea'] > 500)  & (df['GarageArea'] <= 600),  'GarageArea'] = 5,
    df.loc[(df['GarageArea'] > 600)  & (df['GarageArea'] <= 700),  'GarageArea'] = 6,
    df.loc[(df['GarageArea'] > 700)  & (df['GarageArea'] <= 800), 'GarageArea'] = 7,
    df.loc[ df['GarageArea'] > 800, 'GarageArea'] = 8

    return df

def has_pool(df):
    """
        count    1460.000000
        mean        2.758904
        std        40.177307
        min         0.000000
        25%         0.000000
        50%         0.000000
        75%         0.000000
        max       738.000000
    """

    df.loc[ df['PoolArea'] == 0, 'PoolArea'] = 0
    df.loc[ df['PoolArea'] > 0, 'PoolArea'] = 1

    return df

