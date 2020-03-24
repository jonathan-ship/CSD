import numpy as np  # ver 1.18.1
import pandas as pd  # ver 1.0.2
import scipy.stats as st  # scipy ver 1.4.1
import statsmodels as sm  # ver0.11.0
import matplotlib  # ver 3.1.3
import matplotlib.pyplot as plt
import warnings
import time
from datetime import datetime


##TEST test


########## 함수 정의 부분 #############

###########################  1. 데이터 읽기 및 전처리  ########################

def set_data(csv_name, start_dates, final_dates,
             time_range=None, del_nan=True, del_inconsistency=True):
    ### 파일 읽기
    # pd.read_csv
    original_df = pd.read_csv('./' + csv_name, encoding='utf-8')

    ##작업하고자 하는 프로세스의 시작일, 종료일만 담은 DataFrame 만들기
    df_dates = pd.DataFrame(original_df[start_dates + final_dates], copy=True)

    ## start_dates와 final_dates 목록에 중복이 있을 시 start로 쓸 것인지 final로 쓸 것인지 구분해주는 기호 추가
    for i in range(len(start_dates)):
        start_dates[i] += '_S'
    for i in range(len(final_dates)):
        final_dates[i] += '_F'

    df_dates.columns = start_dates + final_dates

    ### 전처리 시작
    ## int -> datetime 전환
    if type(df_dates.iloc[0][0]) == np.int64:
        for col in df_dates:
            df_dates[col] = df_dates[col].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))

    ## 날짜 처리
    ## 날짜 인풋은 time_range를 통해, 시작 날짜와 끝 날짜를 8자리 정수로 받아옴. ex> [20150101, 20171231]
    if time_range != None:
        df_dates.drop(
            df_dates[df_dates[start_dates[0]] < datetime.strptime(str(time_range[0]), '%Y%m%d')].index,
            inplace=True)
        df_dates.drop(
            df_dates[df_dates[start_dates[0]] >= datetime.strptime(str(time_range[1]), '%Y%m%d')].index,
            inplace=True)

    ## nan 처리
    if del_nan == True:
        df_dates.dropna(inplace=True)

    ## inconsistency 처리
    if del_inconsistency == True:
        for SD, FD in zip(start_dates, final_dates):
            df_dates.drop(df_dates[df_dates[SD] > df_dates[FD]].index, inplace=True)

    return df_dates, start_dates, final_dates
    # output인 processed_dates는 DataFrame 형식으로


###########################  2. 기본 계수 계산  ########################

def cal_var(df_dates, process_names, start_dates, final_dates):
    var_table = pd.DataFrame()

    ## te, ce 계산
    # DataFrame[col 1] - DataFrame[col 2] 하면 날짜 차이 자동으로 계산됨
    # .astype('timedelta64[D]').astype(int) 으로 data type 정수로 변환
    #
    # df_te[i] = (df[final_dates[i]] - df[start_dates[i]]).astype('timedelta64[D]').astype(int)
    # 위 예시 참고

    ## 평균, 표준편차는 DataFrame.mean(), DataFrame.std(), np.mean(), np.std() 로 바로 계산 가능

    df_te = pd.DataFrame(columns=process_names)
    for PN, SD, FD in zip(process_names, start_dates, final_dates):
        df_te[PN] = (df_dates[FD] - df_dates[SD]).astype('timedelta64[D]')
    df_te += 1

    ## ra, ca 계산
    ## 이거 계산은 dataframe으로 할 경우 너무 느려서 np 사용 권장

    # 먼저 첫번째 프로세스 시작일 칼럼만 떼어 와서 numpy ndarray로 저장
    # df.vaues 는 df의 값을 ndarray 형태로 반환해줌 (원래 pandas의 dataframe, series 자체가 값을 np.ndarray로 저장함)
    # np.sort 이용해서 정렬
    # iat 계산 및 저장

    # iat 평균 -> ta, 표준편차 계산, ca 계산, ra = 1 / ta 계산

    # 같은 방법으로 마지막 프로세스 끝 칼럼만 떼어와서 rd, cd 계산

    iat = np.zeros((len(df_dates) - 1, len(start_dates + final_dates)))
    np_dates = df_dates.to_numpy(copy=True)
    np_dates.sort(axis=0)

    for i in range(len(start_dates + final_dates)):
        for k in range(len(iat)):
            iat[k][i] = (np_dates[k + 1][i] - np_dates[k][i]).astype('timedelta64[D]') / np.timedelta64(1, 'D')

    temp = np.nanmean(iat, axis=0)
    ta = temp[: len(start_dates)]
    td = temp[len(start_dates):]
    ra = 1 / ta
    rd = 1 / td

    ra[:] = ra[0]

    temp = np.nanstd(iat, axis=0)
    stda = temp[: len(start_dates)]
    stdd = temp[len(start_dates):]
    ca = stda / ta
    cd = stdd / td

    var_table_index = ['Ra', 'Ca', 'Te', 'Ce', 'Rd', 'Cd', 'm', 'u']

    var_table = pd.DataFrame(columns=process_names, index=var_table_index)
    var_table.loc['Ra'] = ra
    var_table.loc['Ca'] = ca
    var_table.loc['Te'] = np.asarray(df_te.mean())
    var_table.loc['Ce'] = np.asarray(df_te.std() / df_te.mean())
    var_table.loc['Rd'] = rd
    var_table.loc['Cd'] = cd

    return var_table, df_te, iat
    # var_table 은 계산 결과 값을 가지는 dataframe.
    # 칼럼명으로 프로세스 이름을, 인덱스로 ra, ca, te, ce, rd, cd, m, u 를 가지는 dataframe 형태

    # te_table은 모든 작업의 작업시간을 가지는 dataframe.
    # 칼럼명으로 프로세스 이름을 가지는 dataframe 형태. 인덱스는 상관 없음

    # iat_table은 첫번째 프로세스의 시작일에서 도착간 간격을 계산한 값. ndarray 형태.


###########################  3. utilization 계산  ########################

## 주어진 m에 따른 u 계산

def cal_u(var_table, set_m):
    # var_table의 변수들과 m으로 공식에 따라 u 계산, var_table에 추가

    ra = var_table.loc[['Ra']].values.flatten()
    te = var_table.loc[['Te']].values.flatten()
    m = set_m

    u = ra * te / m

    var_table.loc['m', :] = np.asarray(m)
    var_table.loc['u', :] = u

    return var_table


## 목표하는 u를 맞추는 m 과 그에 따른 u 계산

def target_u(var_table, set_u):
    # 목표 ra, te, u 이용해서 역으로 m 계산, int로 바꿈

    ra = var_table.loc[['Ra']].values.flatten()
    te = var_table.loc[['Te']].values.flatten()

    temp = np.zeros(len(var_table.columns))
    temp[:] = set_u
    m = ra * te / temp

    m = np.round(m.astype(np.double))

    for i in range(len(m)):
        if m[i] == 0:
            m[i] = 1

    # 위에서 구한 m에 따른 u 다시 계산
    u = ra * te / m

    # m과 u var_table에 추가
    var_table.loc['m', :] = np.asarray(m)
    var_table.loc['u', :] = u

    return var_table


###########################  4. 주요 변수 계산  ########################

def cal_result_table(var_table, process_names):
    # var_table의 변수들로 최종 각 process 별 CT, CTq, WIP, WIPq 계산

    ra = var_table.loc[['Ra']].values.flatten()
    ca = var_table.loc[['Ca']].values.flatten()
    te = var_table.loc[['Te']].values.flatten()
    ce = var_table.loc[['Ce']].values.flatten()
    u = var_table.loc[['u']].values.flatten()
    m = var_table.loc[['m']].values.flatten()

    for i in range(len(var_table.columns) - 1):
        temp = cal_cd(ca=ca[i], ce=ce[i], u=u[i], m=m[i])
        ca[i + 1] = temp

    m = m.astype(np.double)

    CTq = (((ca * ca) + (ce * ce)) / 2) * (np.power(u, np.sqrt(2 * (m + 1)) - 1) / (m * (1 - u))) * te
    CT = CTq + te
    WIPq = CTq * ra
    WIP = CT * ra

    result_table = pd.DataFrame(columns=process_names, index=['CTq', 'CT', 'WIPq', 'WIP'])
    result_table.loc['CTq'] = CTq
    result_table.loc['CT'] = CT
    result_table.loc['WIPq'] = WIPq
    result_table.loc['WIP'] = WIP

    return result_table
    # result_table은 칼럼으로 각 프로세스 명을, 인덱스로 CT, CTq, WIP, WIPq 을 갖는 데이터프레임으로.


def cal_cd(ca, ce, u, m):
    cd = 1 + (1 - u * u) * (ca * ca - 1) + (u * u) * (ce * ce - 1) / np.sqrt(m)
    cd = np.sqrt(cd)
    return cd


def cal_result(te, ce, ra, ca, u, m):
    CTq = (((ca * ca) + (ce * ce)) / 2) * (np.power(u, np.sqrt(2 * (m + 1)) - 1) / (m * (1 - u))) * te
    CT = CTq + te
    WIPq = CTq * ra
    WIP = CT * ra

    return CTq, CT, WIPq, WIP


##########################  6. 분포 fitting 함수  ######################
def best_fit_distribution(data, factor='sse', bins=None, ax=None):
    if bins == None:
        bins = int(np.round(data.max() - data.min()))

    #     print(bins)

    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [
        st.beta,
        st.chi2,
        st.expon,
        st.exponnorm,
        st.f,
        st.gamma,
        st.logistic,
        st.lognorm,
        st.norm,
        st.pareto,
        st.triang,
        st.uniform,
        st.weibull_min
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_loglh = - np.inf
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                loglh = distribution.logpdf(data, arg, loc, scale).sum()
                sse = np.sum(np.power(y - pdf, 2.0))

                #                 print(distribution, loglh, sse)

                # identify if this distribution is better
                if factor == 'loglh':
                    if best_loglh < loglh:
                        best_distribution = distribution
                        best_params = params
                        best_loglh = loglh
                elif factor == 'sse':
                    if best_sse > sse:
                        best_distribution = distribution
                        best_params = params
                        best_sse = sse
                else:
                    break

        except Exception:
            pass

    return (best_distribution.name, best_params)


def searcher(dist, params, target_value, step_size, tunning_param='scale', _round=True):
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    if tunning_param == 'scale':
        origin_param = scale
    else:
        print("Select valid tunning parameter")

    count = 0
    while True:

        errors = np.empty(9)
        errors[:] = np.inf

        if _round == True:
            for i in range(9):
                temp_param = origin_param + step_size * (i - 4)
                #                 print(arg, loc, temp_param)
                try:
                    temp = dist.rvs(arg, loc, temp_param, size=1000000)
                    temp = temp[temp > 0]
                    temp = np.round(temp)
                    #                     print(target_value, temp.mean())
                    errors[i] = np.absolute(target_value - temp.mean())
                except:
                    #                     print('out' + str(i))
                    pass

        else:
            for i in range(9):
                temp_param = origin_param + step_size * (i - 4)
                #                 print(arg, loc, temp_param)
                try:
                    temp = dist.rvs(arg, loc, temp_param, size=1000000)
                    temp = temp[temp > 0]
                    #                     temp = np.round(temp)
                    #                     print(target_value, temp.mean())
                    errors[i] = np.absolute(target_value - temp.mean())
                except:
                    #                     print('out' + str(i))
                    pass

        tunned_param_index = errors.argmin()
        tunned_param = origin_param + step_size * (tunned_param_index - 4)
        tunned_error = errors.min()

        origin_param = tunned_param

        #         print(errors)
        #         print("tunned_param_index : ", tunned_param_index)

        if tunned_param_index != 0 and tunned_param_index != 8:
            break

        count += 1
        if count == 10:
            print("Went out of range, fitting failed")
            break
    #     print(tunned_param)
    print("Error :", tunned_error)
    return tunned_param, tunned_error


def distribution_tunning(data, dist, params, accuracy=0.1, _round=True):
    # target 값 설정
    target = data.mean()

    ## 튜닝 시작

    # scale tunning
    origin_scale = params[-1]

    step_size = 1
    for i in range(5):
        temp_scale, tunned_error = searcher(dist, params, target, step_size=step_size, _round=_round)

        temp_list = list(params)
        temp_list[-1] = temp_scale
        params = tuple(temp_list)

        step_size *= 0.1
        if tunned_error < accuracy:
            break

    return dist, params


def distribution_finder(table, quantile=None, factor='sse',
                        tunning=True, accuracy=0.01, _round=True):
    temptime = time.time()

    if type(table) == pd.DataFrame:
        pass
    elif type(table) == np.ndarray:
        table = pd.DataFrame(table)
    elif type(table) == pd.Series:
        table = pd.DataFrame(table)

    dist_list = []
    params_list = []
    str_list = []
    result_list = []
    cv_list = []

    for i, col in zip(range(len(table)), table):
        data = pd.Series(table[col], copy=True)

        if quantile != None:
            data = data[data <= data.quantile(quantile)]

        best_fit_name, best_fit_params = best_fit_distribution(data, factor=factor)
        best_dist = getattr(st, best_fit_name)

        if tunning == True:
            print("Distribution " + str(i) + " is tunning...")
            best_dist, best_params = distribution_tunning(data, best_dist, best_fit_params, accuracy=accuracy,
                                                          _round=_round)

        param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, best_fit_params)])
        dist_str = '{}({})'.format(best_fit_name, param_str)

        params = best_params
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        test = best_dist.rvs(arg, loc, scale, size=1000000)
        test = test[test > 0]
        if _round == True:
            test = np.round(test)
        test = pd.Series(test)

        result_list.append(pd.DataFrame({'Original': data.describe(), 'Fitting': test.describe()}))
        cv_list.append("\nCV"
                       + "\nOriginal : " + str(round(data.std() / data.mean(), 2))
                       + "    Fitting : " + str(round(test.std() / test.mean(), 2)))

        dist_list.append(best_dist)
        params_list.append(best_fit_params)
        str_list.append(dist_str)

    print("\n\n\nFound distributions are :")
    for i, str_ in zip(range(len(str_list)), str_list):
        print(str(i) + " : " + str_)

    print("\n\nComparing original data with fitted distribution")
    for i, result, cv in zip(range(len(result_list)), result_list, cv_list):
        print(str(i) + " : ")
        print(result)
        print(cv)
        print('\n\n')

    print("Process finished in " + str(time.time() - temptime) + " sec")

    return dist_list, params_list



class CSD_Calculator:

    def __init__(self, csv_name, start_dates, final_dates, set_m=None, set_u=None,
                 process_names=None, time_range=None, del_nan=True, del_inconsistency=True):
        #         self.csv_name = csv_name
        #         self.start_dates = start_dates
        #         self.final_dates = final_dates
        self.process_names = []
        if process_names == None:
            for i in range(len(start_dates)):
                temp = 'process' + str(i)
                self.process_names.append('temp')
        else:
            self.process_names = process_names

        ## 1. 데이터 읽기 및 전처리
        # set_data
        # Input : 파일명, 시작/종료 날짜, 시간 범위, 전처리 옵션
        # Output : 전처리 완료된 날짜 Table

        self.df_dates, self.start_dates, self.final_dates = set_data(csv_name, start_dates, final_dates,
                                                                     time_range=time_range,
                                                                     del_nan=del_nan,
                                                                     del_inconsistency=del_inconsistency)

        ## 2. 기본 계수 계산
        # cal_var
        # Input : 전처리 완료된 날짜 Table, 프로세스 이름
        # Output : 프로세스 별 계산된 계수 Table (var_table) , Te Table, IAT Table

        self.var_table, self.df_te, self.iat = cal_var(self.df_dates, self.process_names, self.start_dates,
                                                       self.final_dates)

        ## 3. utilization 계산
        # cal_u - 입력받은 m에 해당하는 u 계산
        # Input : var_table, m
        # Output : var_table 에 m 추가

        # set_u - 입력받은 목표 u를 달성하는 m과 그에 따른 실제 u 계산
        # Input : var_table, u
        # Output : var_table 에 u와 m 추가

        if set_m != None:
            self.var_table = cal_u(self.var_table, set_m)

        elif set_u != None:
            self.var_table = target_u(self.var_table, set_u)

        else:
            print("Set m or set u")

        ## 4. 주요 Factor 계산
        # cal_result_table
        # Input : var_table
        # Output : result_table ( CT, CTq, WIP, WIPq )

        self.result_table = cal_result_table(self.var_table, self.process_names)

        ## 5. 민감도 분석

        ## 6. 분포 Fitting 함수
        # dist_fit
        # Input : te_table, iat_table
        # Output : dist

        ## 7. 각 작업의 일별/월별/분기별 작업량 확인 함수
        # num_act
        # input : activity, span
        # output : graph

    # def distribution_finder(self, table,
