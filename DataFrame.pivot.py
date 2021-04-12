import pandas as pd
df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two',
                           'two'],
                   'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'baz': [1, 2, 3, 4, 5, 6],
                   'zoo': ['x', 'y', 'z', 'q', 'w', 't']})

df2=pd.DataFrame([['one','A',1,'x'],
                  ['one','B',2,'y'],
                  ['one','C',3,'z'],
                  ['two','A',4,'q'],
                  ['two','B',5,'w'],
                  ['two','C',6,'t'],
                  ['two','C',6,'s']], columns=['foo','bar','baz','zoo'])

print(df)
print(df2)

df_pivot=df.pivot(index='foo', columns='bar', values='zoo')
print(df_pivot)
print(df_pivot.index)

df_copy=df.copy()

# df2_pivot=df2.pivot(index='foo', columns='bar', values='zoo')
# print(df2_pivot)
#=> cannot reshape df if index contains duplicate entriess

df_to_change_column_name=pd.DataFrame({'id':['a','b','c','d'], 'name':['park','jeong','kim','hong']})
df_to_change_column_name.rename(columns={'id':'x_id'},inplace=True)
