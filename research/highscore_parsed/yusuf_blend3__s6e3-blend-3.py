# %% cell 1
import numpy as np
import pandas as pd

import ast,shutil,copy
import warnings
from bokeh.plotting import figure, gridplot 
from bokeh.io import output_file, show, output_notebook; output_notebook()

warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %% cell 2
def bokeh_show(
        params,
        df_cross,
        show_figures1, 
        show_figures2, wps_fig2,
        color_cross):

    colors = [subm['color'] for subm in params['subm']]
    
    def dossier(js,subms,cols):
        def quant(i,js,subms,cols):
            return {"c" : i, "q" : sum([1 for subm in cols[i] if subm == subms[js]])}
        return {
            'name' : subms[js],
            'q_in' : [quant(i,js,subms,cols) for i in range(len(subms))]
        }
    alls = pd.read_csv(f'tida_desc.csv')
    matrix = [ast.literal_eval(str(row.alls)) for row in alls.itertuples()]
    subms = sorted(matrix[0])
    cols = [[data[i] for data in matrix] for i in range(len(subms))]
    df_subms = pd.DataFrame({f'col_{i}': [x[i] for x in matrix] for i in range(len(subms))})
    dossiers = [dossier(js,subms,cols) for js in range(len(subms))]
    subm_names = [one_dossier['name'] for one_dossier in dossiers]
    figures1,qss,i = [],[],0
    height = 100 if len(colors)==2\
        else 134 if len(colors)==3 else (154 if len(colors)==4 else 174)
    for one_dossier in dossiers: 
        i_col = 'alls. ' + str(one_dossier['q_in'][i]['c'])
        qs = [one['q'] for one in one_dossier['q_in']]
        x_names = [name.replace("Group","").replace("subm_","") for name in subm_names]
        width = 140
        f = figure(x_range=x_names,width=width, height=height, title=i_col)
        f.vbar(x=x_names, width=0.585, top=qs, color=colors)
        figures1.append(f)
        qss.append(qs)
        i+=1
    grid = gridplot([figures1])
    output_file('tida_alls.html')
    if show_figures1 == True: show(grid)
    sub_wts = params['subwts']
    main_wts = [subm['weight'] for subm in params['subm']]
    mms,acc_mass = [],[]
    for j in range(len(dossiers)):
        one_dossier = dossiers[j]
        qs = [one['q'] for one in one_dossier['q_in']]
        mm = [qs[h] * (main_wts[j] + sub_wts[h]) for h in range(len(sub_wts))]
        mass = sum(mm)
        mms.append(mm)
        acc_mass.append(round(mass))                        #subm_names[::-1]
    y_names = [name + " - " + str(mass) for name,mass in zip(subm_names,acc_mass)]
    f1 = figure(y_range=y_names, width=270, height=height, title='relations of general masses')
    f1.hbar(y=y_names, height=0.555, right=acc_mass, left=0, color=colors)
    output_file('tida_alls2.html')
    alls = [f'alls.{i}' for i in range(len(dossiers))]
    subm = [f'sub{i}'   for i in range(len(dossiers))] 
    mmsT  = np.asarray(mms).T
    data = {'cols' : alls}
    for i in range(len(dossiers)): data[f'sub{i}'] = mmsT[i,:]
    f2 = figure(y_range=alls, height=height, width=270, title="relations of columns masses")
    f2.hbar_stack(subm, y='cols', height=0.555, color=colors, source=data)
    qssT  = np.asarray(qss).T
    data = {'cols' : alls}
    for i in range(len(dossiers)): data[f'sub{i}'] = qssT[i,:]
    f3 = figure(y_range=alls, height=height, width=245, title="ratios in columns")
    f3.hbar_stack(subm, y='cols', height=0.555, color=colors, source=data)
    grid = gridplot([[f3,f2,f1]])
    show(grid)
    if show_figures2 == True:
        def read(params,i):
            FiN = params["path"] + params["subm"][i]["name"] + ".csv"
            target_name_back = {'target':params["target"],'pred':params["target"]}
            return pd.read_csv(FiN).rename(columns=target_name_back)
        dfs = [read(params,i) for i in range(len(params["subm"]))] + [df_cross]
        _height = 358 if len(params["subm"]) == 11 else 254
        f   = figure(width=785, height=_height)
        f.title.text = 'Click on legend entries to mute the corresponding lines'
        b,e        = 21000,21154
        line_x     = [dfs[i][b:e]['id']         for i in range(len(dfs))]
        line_y     = [dfs[i][b:e]['exam_score'] for i in range(len(dfs))]
        color      = colors + [color_cross]
        alpha      = [0.8 for i in range(len(dfs)-1)] + [0.95]
        lws        = [1.0 for i in range(len(dfs)-1)] + [1.00]
        legend = subm_names + ['cross']
        for i in range(len(legend)):
            f.line(line_x[i], line_y[i], line_width=lws[i], color=color[i], alpha=alpha[i],
                   muted_color='white',legend_label=legend[i])
        f.legend.location = "top_left"
        f.legend.click_policy="mute"
        show(f)

# %% cell 3
def matrix_vs(path,fs_names):
    def load(path,fs_names):
        dfs = [pd.read_csv(path + name_subm +'.csv') for name_subm in fs_names]
        for i in range(len(dfs)):
            dfs[i] = dfs[i].rename(columns={"Churn": f'{fs_names[i]}'})
        dfsm = pd.merge(dfs[0], dfs[1], on="id")
        for i in range(2,len(dfs)):
            dfsm = pd.merge(dfsm,dfs[i],on='id')
        return dfsm   
    def make_list_vs(fs_names):
        list = []
        for i in range(0,len(fs_names)-1):
            for j in range(i+1,len(fs_names)):
                list.append(fs_names[i] + "_vs_" + fs_names[j])
        return list
    def get_mvs(dfs, list_vs):
        def get_abs_distance(x,t1,t2):
            return abs(x[t1]-x[t2])
        for vs in list_vs:
            t = vs.split('_vs_')
            dfs[vs] = dfs.apply(lambda x: get_abs_distance(x,t[0],t[1]), axis=1)
        return dfs   
    def distance_vs(name, st_names, list_vs, dfs):
        distances = []
        for st in st_names:
            vs_between = name + "_vs_" + st
            if vs_between not in list_vs:
                distances.append(0)
            else: distances.append(round(dfs[vs_between].sum()))
        return distances
    dfs = load(path,fs_names)
    list_vs = make_list_vs(fs_names)
    mvs = get_mvs(dfs, list_vs)
    m1 = pd.DataFrame({'subm':fs_names})
    m2 = pd.DataFrame({ name :distance_vs(name, fs_names, list_vs, mvs) for name in fs_names})
    matrix = pd.concat([m1,m2],axis=1)
    return matrix


def seaborn_Show(params,file_name_cross=''):
    import matplotlib.pyplot as plt, seaborn as sns
    import warnings; warnings.filterwarnings('ignore')
    plt.figure(figsize=(8.7, 2))
    for subm in params['subm']:
        pred = pd.read_csv(params['path']+subm['name']+'.csv')[params['id_target'][1]]
        sns.kdeplot(pred, label = subm['name'], linewidth = 0.5)
    if file_name_cross != '':
        pred = pd.read_csv(file_name_cross)[params['id_target'][1]]
        sns.kdeplot(pred, label = 'blend', linewidth = 1, linestyle = 'dashed')
    plt.title("KDE")
    plt.xlabel("target")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def display_distances(params):
    files = [subm['name'] for subm in params['subm']]
    distances = matrix_vs ( params['path'], files )            
    display(distances)


def arr_colors(color):
    dskb,mvr = 'deepskyblue','mediumvioletred'
    sg = ['darkgray','silver','gainsboro']
    if color=='red'   or color=='R': return ['firebrick','red','crimson','tomato']     + sg
    if color=='Red'   or color=='r': return ['red','tomato','crimson']                 + sg
    if color=='Green' or color=='G': return ['darkgreen','limegreen','green','lime']   + sg
    if color=='Blue'  or color=='B': return ['midnightblue','blue','mediumblue',dskb]  + sg
    if color=='RGB'   or color=='S': return ['mediumblue','darkgreen','crimson']       + sg
    if color=='RGBM'  or color=='M': return [mvr,'darkorchid','darkmagenta','magenta'] + sg
    return ['black','dimgray','gray'] + sg


def convert(schema):
    colors = arr_colors(schema[2])
    dicts  = [
        {'name': schema[0][i],'weight':schema[1][i],'color':colors[i]} 
        for i in range(len(schema[0]))
    ]
    return {'subm':dicts}

# %% cell 4
def h_blend(
        params, _update={},
        cross='silver',
        details=True,
        fig1=True, fig2=True, wf2=555, 
        dtls=True, dist=True, subm=''):

    if isinstance(params, list): params = convert(params)

    if 'path' in _update or 'subwts' in _update : params.update(_update)
    
    color_cross, dk  = cross, copy.deepcopy(params)

    if details == True:
        dist = True
        show_details,show_figures1,show_figures2 = True,True,True
    else:
        show_details,show_figures1,show_figures2 = dtls,fig1,fig2
        
    file_short_names = [subm['name'] for subm in params['subm']]
    type_sort    = params['type_sort'][0]
    dk['asc']    = params['type_sort'][1]
    dk['desc']   = params['type_sort'][2]
    dk['id']     = params['id_target'][0]
    dk['target'] = params['id_target'][1]

    def read(dk,i):
        tnm = dk["subm"][i]["name"]
        FiN = dk["path"] + tnm + ".csv"
        return pd.read_csv(FiN).rename(columns={
            'target':tnm, 'pred':tnm, dk["target"]:tnm})
        
    def merge(dfs_subm):
        df_subms = pd.merge(dfs_subm[0],  dfs_subm[1], on=[dk['id']])
        for i in range(2, len(dk["subm"])): 
            df_subms = pd.merge(df_subms, dfs_subm[i], on=[dk['id']])
        return df_subms
        
    def da(dk,sorting_direction,show_details):
        
        df_subms = merge([read(dk,i) for i in range(len(dk["subm"]))])
        cols = [col for col in df_subms.columns if col != dk['id']]
        short_name_cols = [c for c in cols]
        
        def alls1(x, sd=sorting_direction,cs=cols):
            reverse = True if sd=='desc' else False
            tes = {c: x[c] for c in cs}.items()
            subms_sorted = [t[0] for t in sorted(tes,key=lambda k:k[1],reverse=reverse)]
            return subms_sorted

        import random

        def alls2(x, sd=sorting_direction,cs=cols):
            reverse = True if sd=='desc' else False
            tes = {c: x[c] for c in cs}.items()
            subms_random = [t[0] for t in tes]
            random.shuffle(subms_random)
            return subms_random

        alls = alls1 if type_sort == 'asc/desc' else alls2
            
        def summa(x,cs,wts,ic_alls): 
            return sum([x[cs[j]] * (wts[0][j] + wts[1][ic_alls[j]]) for j in range(len(cs))])
            
        wts = [[[e['weight'] for e in dk["subm"]], [w for w in dk["subwts"]]]]
          
        def correct(x, cs=cols, wts=wts):
            i = [x['alls'].index(c) for c in short_name_cols]
            return summa(x,cs,wts[0],i)

        if len(wts) == 1:
            correct_sub_weights = [wt for wt in dk["subwts"]]
            weights = [subm['weight'] for subm in dk["subm"]]
            def correct(x, cs=cols, w=weights, cw=correct_sub_weights):
                ic = [x['alls'].index(c) for c in short_name_cols]
                cS = [x[cols[j]] * (w[j] + cw[ic[j]]) for j in range(len(cols))]
                return sum(cS)
                
        if len(wts) > 1 or "subwts2" in dk:

            wts = [
                [[e['weight'] for e in dk["subm"]], [w for w in dk["subwts" ]]],
                [[e['weight'] for e in dk["subm2"]],[w for w in dk["subwts2"]]],
                [[e['weight'] for e in dk["subm3"]],[w for w in dk["subwts3"]]],
            ]

            def correct(x, cs=cols, wts=wts):
                i = [x['alls'].index(c) for c in short_name_cols]
                if   0.0540 < x['mx-m'] <= 0.0740: return summa(x,cs,wts[2],i)
                if   0.0000 < x['mx-m'] <= 0.0050: return summa(x,cs,wts[1],i)
                else:                              return summa(x,cs,wts[0],i)
                   
        def amxm(x, cs=cols):
            list_values = x[cs].to_list()
            mxm = abs(max(list_values)-min(list_values))
            return mxm

        if len(wts) > 1 or "subwts2" in dk:
            df_subms['mx-m']   = df_subms.apply(lambda x: amxm   (x), axis=1)
        df_subms['alls']       = df_subms.apply(lambda x: alls   (x), axis=1)
        df_subms[dk["target"]] = df_subms.apply(lambda x: correct(x), axis=1)
        schema_rename = { old_nc:new_shnc for old_nc, new_shnc in zip(cols, short_name_cols) }
        df_subms = df_subms.rename(columns=schema_rename)
        df_subms = df_subms.rename(columns={dk["target"]:"ensemble"})
        df_subms.insert(loc=1, column=' _ ', value=['   '] * len(df_subms))
        df_subms[' _ '] = df_subms[' _ '].astype(str)
        pd.set_option('display.max_rows',100)
        pd.set_option('display.float_format', '{:.5f}'.format)
        if len(wts) > 1: 
            vcols = [dk['id']] + [' _ '] + short_name_cols + [' _ '] + ['mx-m'] + [' _ '] +\
                      ['alls'] + [' _ '] + ['ensemble']
        else:
            vcols = [dk['id']] + [' _ '] + short_name_cols + [' _ '] +\
                      ['alls'] + [' _ '] + ['ensemble']
        df_subms = df_subms[vcols]
        if show_details and sorting_direction=='desc': display(df_subms.head(5))
        pd.set_option('display.float_format', '{:.5f}'.format)
        df_subms = df_subms.rename(columns={"ensemble":dk["target"]})
        if sorting_direction=='desc': 
            df_subms.to_csv(f'tida_{sorting_direction}.csv', index=False)
        return df_subms[[dk['id'],dk['target']]]
   
    def ensemble_da(dk,        show_details): 
        dfD    = da(dk,'desc', show_details)
        dfA    = da(dk,'asc',  show_details)
        dfA[dk['target']] = dk['desc']*dfD[dk['target']] + dfA[dk['target']]*dk['asc']
        return dfA

    da = ensemble_da(dk,show_details)

    if subm != '': da.to_csv(subm, index=False)
        
    return  da

# %% cell 5
def shutil_copy(inst,dest):
    shutil.copy(inst, dest)
    return pd.read_csv(dest)


def b2(fin0,fin1,wts,out):
    df = pd.read_csv('/kaggle/input/competitions/playground-series-s6e3/sample_submission.csv')
    df0 = pd.read_csv(path + fin0 + '.csv')
    df1 = pd.read_csv(path + fin1 + '.csv')
    df['Churn'] = \
        df0['Churn'] * wts[0] + df1['Churn'] * wts[1]
    df.to_csv(out,index=False)                  
    return df


# Ensure these paths in your b2 function match your input list
def b2(df0, wts, df1, subm='submission.csv'):
    df = pd.read_csv('/kaggle/input/competitions/playground-series-s6e3/sample_submission.csv')
    df['Churn'] = df0['Churn'] * wts[1] + df1['Churn'] * wts[0]
    df.to_csv(subm, index=False)


def para(df_Main, weights, dfs_Aux, params_Aux):
    for i in range(len(dfs_Aux)):
        b2(df_Main, weights, dfs_Aux[i], params_Aux['subm'][i]['name']+'.csv')
    return copy.deepcopy(params_Aux)

# %% cell 6
# weights1,weights2,weights3 = [0.96,0.04], [0.89,0.11], [0.82,0.18]
# Mixing ratios — give Main more influence
weights1, weights2, weights3 = [0.70, 0.30], [0.60, 0.40], [0.50, 0.50]

ct1,ct2 = 1.00118, 1.00118

# %% cell 8
A = pd.read_csv('/kaggle/input/notebooks/yusufmurtaza01/s6e3-blend-2/submission.csv')
B = pd.read_csv('/kaggle/input/notebooks/yusufmurtaza01/s6e3-blending/submission.csv')
C = pd.read_csv('/kaggle/input/notebooks/anthonytherrien/predict-customer-churn-blend/submission.csv')
D = pd.read_csv('/kaggle/input/notebooks/dmahajanbe23/customer-churn-blend-0-91691/submission.csv')


A.to_csv('A.csv', index=False)
B.to_csv('B.csv', index=False)
C.to_csv('C.csv', index=False)
D.to_csv('D.csv', index=False)

# %% cell 9
params_Main = {
      'path'     : f'/kaggle/working/', 
      'id_target': ['id',"Churn"],
      'type_sort': ['asc/desc', 0.30, 0.70],
      'subm'     : [
          {'name': 'A', 'weight': +0.35, 'color': 'darkmagenta'}, # Best file
          {'name': 'B', 'weight': +0.30, 'color': 'darkgreen'},
          {'name': 'C', 'weight': +0.20, 'color': 'deeppink'},
          {'name': 'D', 'weight': +0.15, 'color': 'magenta'},
      ]
}

# %% cell 11
params_Aux = {
      'path'     : f'/kaggle/working/',
      'id_target': ['id',"Churn"],        
      'type_sort': ['asc/desc', 0.30, 0.70],
      'subwts'   : [-0.25, 0.00, +0.55, -0.30],
      'subm'     : [
          {'name': 'Main+a', 'weight': +0.48, 'color': 'red'  },
          {'name': 'Main+b', 'weight': +0.32, 'color': 'brown'},
          {'name': 'Main+c', 'weight': +0.08, 'color': 'navy' },
          {'name': 'Main+d', 'weight': +0.12, 'color': 'black'},
      ]
}

# %% cell 12
a = pd.read_csv('/kaggle/input/notebooks/yusufmurtaza01/s6e3-xgb/submissions/submission_XGB_FULL_FE_s1234.csv')
b = pd.read_csv('/kaggle/input/notebooks/yusufmurtaza01/s6e3-xgb/submissions/submission_XGB_FULL_FE_s999.csv')
c = pd.read_csv('/kaggle/input/notebooks/yusufmurtaza01/s6e3-xgb/submissions/submission_XGB_FULL_FE_s7.csv')
d = pd.read_csv('/kaggle/input/notebooks/yusufmurtaza01/s6e3-xgb/submissions/submission_XGB_FULL_FE_s42.csv')

a.to_csv('a.csv', index=False)
b.to_csv('b.csv', index=False)
c.to_csv('c.csv', index=False)
d.to_csv('d.csv', index=False)

dfs_Aux = [c, b, a, d]

# %% cell 13
df_Main = h_blend(params_Main,_update={'subwts':[+0.55, -0.10, -0.20, -0.25]})

df1 = h_blend( para(df_Main, weights1, dfs_Aux, params_Aux) )

df_Main = h_blend(params_Main,_update={'subwts':[+0.11, -0.01, -0.03, -0.07]})

df2 = h_blend( para(df_Main, weights2, dfs_Aux, params_Aux) )

df_Main = h_blend(params_Main,_update={'subwts':[+0.55, -0.10, -0.20, -0.25]})

df3 = h_blend( para(df_Main, weights3, dfs_Aux, params_Aux) )

# %% cell 14
seaborn_Show(params_Aux)

# %% cell 15
df1.rename(columns={'Churn':'es1'},inplace=True)
df2.rename(columns={'Churn':'es2'},inplace=True)
df3.rename(columns={'Churn':'es3'},inplace=True)

df = pd.merge(df1,df2,on='id')
df = pd.merge(df, df3,on='id')


def trend(x):
    e1,e2,e3 = x['es1'],x['es2'],x['es3']
    if e1 < e3 and e2 < e3: return x['es3'] * (ct1 - 0.0001*(e3-e1))  
    if e1 > e2 and e2 > e3: return x['es3'] / (ct2 - 0.0001*(e1-e3))
    return x['es3']


df['Churn'] = df.apply(lambda x: trend(x), axis=1)

df

# %% cell 16
for name in 'a,b,c,d'.split(','):
    file = f'/kaggle/working/Main+2{name}.csv'
    if os.path.isfile(file): os.remove(file)

# %% cell 17
df[['id','Churn']].to_csv('submission.csv',index=False)
df[['id','Churn']]

