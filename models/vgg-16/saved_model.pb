
А§
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*	2.0.0-rc02v2.0.0-beta1-5101-gc75bb668ЇЌ

block1_conv1/kernelVarHandleOp*$
shared_nameblock1_conv1/kernel*
dtype0*
_output_shapes
: *
shape:@

'block1_conv1/kernel/Read/ReadVariableOpReadVariableOpblock1_conv1/kernel*
dtype0*&
_output_shapes
:@
z
block1_conv1/biasVarHandleOp*
shape:@*"
shared_nameblock1_conv1/bias*
dtype0*
_output_shapes
: 
s
%block1_conv1/bias/Read/ReadVariableOpReadVariableOpblock1_conv1/bias*
dtype0*
_output_shapes
:@

block1_conv2/kernelVarHandleOp*
shape:@@*$
shared_nameblock1_conv2/kernel*
dtype0*
_output_shapes
: 

'block1_conv2/kernel/Read/ReadVariableOpReadVariableOpblock1_conv2/kernel*
dtype0*&
_output_shapes
:@@
z
block1_conv2/biasVarHandleOp*
shape:@*"
shared_nameblock1_conv2/bias*
dtype0*
_output_shapes
: 
s
%block1_conv2/bias/Read/ReadVariableOpReadVariableOpblock1_conv2/bias*
dtype0*
_output_shapes
:@

block2_conv1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*$
shared_nameblock2_conv1/kernel

'block2_conv1/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1/kernel*
dtype0*'
_output_shapes
:@
{
block2_conv1/biasVarHandleOp*
_output_shapes
: *
shape:*"
shared_nameblock2_conv1/bias*
dtype0
t
%block2_conv1/bias/Read/ReadVariableOpReadVariableOpblock2_conv1/bias*
dtype0*
_output_shapes	
:

block2_conv2/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:*$
shared_nameblock2_conv2/kernel

'block2_conv2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2/kernel*(
_output_shapes
:*
dtype0
{
block2_conv2/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*"
shared_nameblock2_conv2/bias
t
%block2_conv2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2/bias*
dtype0*
_output_shapes	
:

block3_conv1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:*$
shared_nameblock3_conv1/kernel

'block3_conv1/kernel/Read/ReadVariableOpReadVariableOpblock3_conv1/kernel*
dtype0*(
_output_shapes
:
{
block3_conv1/biasVarHandleOp*"
shared_nameblock3_conv1/bias*
dtype0*
_output_shapes
: *
shape:
t
%block3_conv1/bias/Read/ReadVariableOpReadVariableOpblock3_conv1/bias*
dtype0*
_output_shapes	
:

block3_conv2/kernelVarHandleOp*
shape:*$
shared_nameblock3_conv2/kernel*
dtype0*
_output_shapes
: 

'block3_conv2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv2/kernel*
dtype0*(
_output_shapes
:
{
block3_conv2/biasVarHandleOp*
_output_shapes
: *
shape:*"
shared_nameblock3_conv2/bias*
dtype0
t
%block3_conv2/bias/Read/ReadVariableOpReadVariableOpblock3_conv2/bias*
_output_shapes	
:*
dtype0

block3_conv3/kernelVarHandleOp*
shape:*$
shared_nameblock3_conv3/kernel*
dtype0*
_output_shapes
: 

'block3_conv3/kernel/Read/ReadVariableOpReadVariableOpblock3_conv3/kernel*
dtype0*(
_output_shapes
:
{
block3_conv3/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*"
shared_nameblock3_conv3/bias
t
%block3_conv3/bias/Read/ReadVariableOpReadVariableOpblock3_conv3/bias*
dtype0*
_output_shapes	
:

block4_conv1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:*$
shared_nameblock4_conv1/kernel

'block4_conv1/kernel/Read/ReadVariableOpReadVariableOpblock4_conv1/kernel*
dtype0*(
_output_shapes
:
{
block4_conv1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*"
shared_nameblock4_conv1/bias
t
%block4_conv1/bias/Read/ReadVariableOpReadVariableOpblock4_conv1/bias*
dtype0*
_output_shapes	
:

block4_conv2/kernelVarHandleOp*$
shared_nameblock4_conv2/kernel*
dtype0*
_output_shapes
: *
shape:

'block4_conv2/kernel/Read/ReadVariableOpReadVariableOpblock4_conv2/kernel*
dtype0*(
_output_shapes
:
{
block4_conv2/biasVarHandleOp*
shape:*"
shared_nameblock4_conv2/bias*
dtype0*
_output_shapes
: 
t
%block4_conv2/bias/Read/ReadVariableOpReadVariableOpblock4_conv2/bias*
_output_shapes	
:*
dtype0

block4_conv3/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:*$
shared_nameblock4_conv3/kernel

'block4_conv3/kernel/Read/ReadVariableOpReadVariableOpblock4_conv3/kernel*
dtype0*(
_output_shapes
:
{
block4_conv3/biasVarHandleOp*
shape:*"
shared_nameblock4_conv3/bias*
dtype0*
_output_shapes
: 
t
%block4_conv3/bias/Read/ReadVariableOpReadVariableOpblock4_conv3/bias*
dtype0*
_output_shapes	
:

block5_conv1/kernelVarHandleOp*$
shared_nameblock5_conv1/kernel*
dtype0*
_output_shapes
: *
shape:

'block5_conv1/kernel/Read/ReadVariableOpReadVariableOpblock5_conv1/kernel*
dtype0*(
_output_shapes
:
{
block5_conv1/biasVarHandleOp*
_output_shapes
: *
shape:*"
shared_nameblock5_conv1/bias*
dtype0
t
%block5_conv1/bias/Read/ReadVariableOpReadVariableOpblock5_conv1/bias*
dtype0*
_output_shapes	
:

block5_conv2/kernelVarHandleOp*
_output_shapes
: *
shape:*$
shared_nameblock5_conv2/kernel*
dtype0

'block5_conv2/kernel/Read/ReadVariableOpReadVariableOpblock5_conv2/kernel*(
_output_shapes
:*
dtype0
{
block5_conv2/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*"
shared_nameblock5_conv2/bias
t
%block5_conv2/bias/Read/ReadVariableOpReadVariableOpblock5_conv2/bias*
dtype0*
_output_shapes	
:

block5_conv3/kernelVarHandleOp*
shape:*$
shared_nameblock5_conv3/kernel*
dtype0*
_output_shapes
: 

'block5_conv3/kernel/Read/ReadVariableOpReadVariableOpblock5_conv3/kernel*
dtype0*(
_output_shapes
:
{
block5_conv3/biasVarHandleOp*"
shared_nameblock5_conv3/bias*
dtype0*
_output_shapes
: *
shape:
t
%block5_conv3/bias/Read/ReadVariableOpReadVariableOpblock5_conv3/bias*
dtype0*
_output_shapes	
:
r

fc1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:
*
shared_name
fc1/kernel
k
fc1/kernel/Read/ReadVariableOpReadVariableOp
fc1/kernel*
dtype0* 
_output_shapes
:

i
fc1/biasVarHandleOp*
shape:*
shared_name
fc1/bias*
dtype0*
_output_shapes
: 
b
fc1/bias/Read/ReadVariableOpReadVariableOpfc1/bias*
dtype0*
_output_shapes	
:
q

fc2/kernelVarHandleOp*
shared_name
fc2/kernel*
dtype0*
_output_shapes
: *
shape:	
j
fc2/kernel/Read/ReadVariableOpReadVariableOp
fc2/kernel*
dtype0*
_output_shapes
:	
h
fc2/biasVarHandleOp*
shape:*
shared_name
fc2/bias*
dtype0*
_output_shapes
: 
a
fc2/bias/Read/ReadVariableOpReadVariableOpfc2/bias*
dtype0*
_output_shapes
:

predictions/kernelVarHandleOp*#
shared_namepredictions/kernel*
dtype0*
_output_shapes
: *
shape
:
y
&predictions/kernel/Read/ReadVariableOpReadVariableOppredictions/kernel*
dtype0*
_output_shapes

:
x
predictions/biasVarHandleOp*
shape:*!
shared_namepredictions/bias*
dtype0*
_output_shapes
: 
q
$predictions/bias/Read/ReadVariableOpReadVariableOppredictions/bias*
dtype0*
_output_shapes
:
x
training/Adam/iterVarHandleOp*#
shared_nametraining/Adam/iter*
dtype0	*
_output_shapes
: *
shape: 
q
&training/Adam/iter/Read/ReadVariableOpReadVariableOptraining/Adam/iter*
dtype0	*
_output_shapes
: 
|
training/Adam/beta_1VarHandleOp*
dtype0*
_output_shapes
: *
shape: *%
shared_nametraining/Adam/beta_1
u
(training/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining/Adam/beta_1*
dtype0*
_output_shapes
: 
|
training/Adam/beta_2VarHandleOp*%
shared_nametraining/Adam/beta_2*
dtype0*
_output_shapes
: *
shape: 
u
(training/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining/Adam/beta_2*
_output_shapes
: *
dtype0
z
training/Adam/decayVarHandleOp*
dtype0*
_output_shapes
: *
shape: *$
shared_nametraining/Adam/decay
s
'training/Adam/decay/Read/ReadVariableOpReadVariableOptraining/Adam/decay*
dtype0*
_output_shapes
: 

training/Adam/learning_rateVarHandleOp*
dtype0*
_output_shapes
: *
shape: *,
shared_nametraining/Adam/learning_rate

/training/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining/Adam/learning_rate*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
_output_shapes
: *
shape: *
shared_nametotal*
dtype0
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
shape: *
shared_namecount*
dtype0*
_output_shapes
: 
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 

training/Adam/fc1/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:
*+
shared_nametraining/Adam/fc1/kernel/m

.training/Adam/fc1/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/fc1/kernel/m*
dtype0* 
_output_shapes
:


training/Adam/fc1/bias/mVarHandleOp*
shape:*)
shared_nametraining/Adam/fc1/bias/m*
dtype0*
_output_shapes
: 

,training/Adam/fc1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/fc1/bias/m*
dtype0*
_output_shapes	
:

training/Adam/fc2/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:	*+
shared_nametraining/Adam/fc2/kernel/m

.training/Adam/fc2/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/fc2/kernel/m*
dtype0*
_output_shapes
:	

training/Adam/fc2/bias/mVarHandleOp*
shape:*)
shared_nametraining/Adam/fc2/bias/m*
dtype0*
_output_shapes
: 

,training/Adam/fc2/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/fc2/bias/m*
dtype0*
_output_shapes
:
 
"training/Adam/predictions/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape
:*3
shared_name$"training/Adam/predictions/kernel/m

6training/Adam/predictions/kernel/m/Read/ReadVariableOpReadVariableOp"training/Adam/predictions/kernel/m*
dtype0*
_output_shapes

:

 training/Adam/predictions/bias/mVarHandleOp*1
shared_name" training/Adam/predictions/bias/m*
dtype0*
_output_shapes
: *
shape:

4training/Adam/predictions/bias/m/Read/ReadVariableOpReadVariableOp training/Adam/predictions/bias/m*
dtype0*
_output_shapes
:

training/Adam/fc1/kernel/vVarHandleOp*+
shared_nametraining/Adam/fc1/kernel/v*
dtype0*
_output_shapes
: *
shape:


.training/Adam/fc1/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/fc1/kernel/v*
dtype0* 
_output_shapes
:


training/Adam/fc1/bias/vVarHandleOp*
shape:*)
shared_nametraining/Adam/fc1/bias/v*
dtype0*
_output_shapes
: 

,training/Adam/fc1/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/fc1/bias/v*
dtype0*
_output_shapes	
:

training/Adam/fc2/kernel/vVarHandleOp*+
shared_nametraining/Adam/fc2/kernel/v*
dtype0*
_output_shapes
: *
shape:	

.training/Adam/fc2/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/fc2/kernel/v*
dtype0*
_output_shapes
:	

training/Adam/fc2/bias/vVarHandleOp*)
shared_nametraining/Adam/fc2/bias/v*
dtype0*
_output_shapes
: *
shape:

,training/Adam/fc2/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/fc2/bias/v*
dtype0*
_output_shapes
:
 
"training/Adam/predictions/kernel/vVarHandleOp*
_output_shapes
: *
shape
:*3
shared_name$"training/Adam/predictions/kernel/v*
dtype0

6training/Adam/predictions/kernel/v/Read/ReadVariableOpReadVariableOp"training/Adam/predictions/kernel/v*
dtype0*
_output_shapes

:

 training/Adam/predictions/bias/vVarHandleOp*
shape:*1
shared_name" training/Adam/predictions/bias/v*
dtype0*
_output_shapes
: 

4training/Adam/predictions/bias/v/Read/ReadVariableOpReadVariableOp training/Adam/predictions/bias/v*
dtype0*
_output_shapes
:
Z
ConstConst*!
valueB"ХрЯТйщТ)\їТ*
dtype0*
_output_shapes
:

NoOpNoOp
Иx
Const_1Const"/device:CPU:0*ёw
valueчwBфw Bнw
н
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer_with_weights-12
layer-18
layer-19
layer-20
layer_with_weights-13
layer-21
layer_with_weights-14
layer-22
layer_with_weights-15
layer-23
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
R
regularization_losses
 	variables
!trainable_variables
"	keras_api
П
#	arguments
$_variable_dict
%_trainable_weights
&_non_trainable_weights
'_callable_losses
(regularization_losses
)	variables
*trainable_variables
+	keras_api
~

,kernel
-bias
._callable_losses
/regularization_losses
0	variables
1trainable_variables
2	keras_api
~

3kernel
4bias
5_callable_losses
6regularization_losses
7	variables
8trainable_variables
9	keras_api
h
:_callable_losses
;regularization_losses
<	variables
=trainable_variables
>	keras_api
~

?kernel
@bias
A_callable_losses
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
~

Fkernel
Gbias
H_callable_losses
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
h
M_callable_losses
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
~

Rkernel
Sbias
T_callable_losses
Uregularization_losses
V	variables
Wtrainable_variables
X	keras_api
~

Ykernel
Zbias
[_callable_losses
\regularization_losses
]	variables
^trainable_variables
_	keras_api
~

`kernel
abias
b_callable_losses
cregularization_losses
d	variables
etrainable_variables
f	keras_api
h
g_callable_losses
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
~

lkernel
mbias
n_callable_losses
oregularization_losses
p	variables
qtrainable_variables
r	keras_api
~

skernel
tbias
u_callable_losses
vregularization_losses
w	variables
xtrainable_variables
y	keras_api


zkernel
{bias
|_callable_losses
}regularization_losses
~	variables
trainable_variables
	keras_api
m
_callable_losses
regularization_losses
	variables
trainable_variables
	keras_api

kernel
	bias
_callable_losses
regularization_losses
	variables
trainable_variables
	keras_api

kernel
	bias
_callable_losses
regularization_losses
	variables
trainable_variables
	keras_api

kernel
	bias
_callable_losses
regularization_losses
	variables
trainable_variables
	keras_api
m
_callable_losses
regularization_losses
	variables
trainable_variables
	keras_api
m
 _callable_losses
Ёregularization_losses
Ђ	variables
Ѓtrainable_variables
Є	keras_api

Ѕkernel
	Іbias
Ї_callable_losses
Јregularization_losses
Љ	variables
Њtrainable_variables
Ћ	keras_api

Ќkernel
	­bias
Ў_callable_losses
Џregularization_losses
А	variables
Бtrainable_variables
В	keras_api

Гkernel
	Дbias
Е_callable_losses
Жregularization_losses
З	variables
Иtrainable_variables
Й	keras_api
Щ
	Кiter
Лbeta_1
Мbeta_2

Нdecay
Оlearning_rate	ЅmА	ІmБ	ЌmВ	­mГ	ГmД	ДmЕ	ЅvЖ	ІvЗ	ЌvИ	­vЙ	ГvК	ДvЛ
 

,0
-1
32
43
?4
@5
F6
G7
R8
S9
Y10
Z11
`12
a13
l14
m15
s16
t17
z18
{19
20
21
22
23
24
25
Ѕ26
І27
Ќ28
­29
Г30
Д31
0
Ѕ0
І1
Ќ2
­3
Г4
Д5

 Пlayer_regularization_losses
Рlayers
regularization_losses
	variables
Сmetrics
trainable_variables
Тnon_trainable_variables
 
 
 
 

 Уlayer_regularization_losses
Фlayers
regularization_losses
 	variables
Хmetrics
!trainable_variables
Цnon_trainable_variables
 
 
 
 
 
 
 
 

 Чlayer_regularization_losses
Шlayers
(regularization_losses
)	variables
Щmetrics
*trainable_variables
Ъnon_trainable_variables
_]
VARIABLE_VALUEblock1_conv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock1_conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

,0
-1
 

 Ыlayer_regularization_losses
Ьlayers
/regularization_losses
0	variables
Эmetrics
1trainable_variables
Юnon_trainable_variables
_]
VARIABLE_VALUEblock1_conv2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock1_conv2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

30
41
 

 Яlayer_regularization_losses
аlayers
6regularization_losses
7	variables
бmetrics
8trainable_variables
вnon_trainable_variables
 
 
 
 

 гlayer_regularization_losses
дlayers
;regularization_losses
<	variables
еmetrics
=trainable_variables
жnon_trainable_variables
_]
VARIABLE_VALUEblock2_conv1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock2_conv1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
@1
 

 зlayer_regularization_losses
иlayers
Bregularization_losses
C	variables
йmetrics
Dtrainable_variables
кnon_trainable_variables
_]
VARIABLE_VALUEblock2_conv2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock2_conv2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

F0
G1
 

 лlayer_regularization_losses
мlayers
Iregularization_losses
J	variables
нmetrics
Ktrainable_variables
оnon_trainable_variables
 
 
 
 

 пlayer_regularization_losses
рlayers
Nregularization_losses
O	variables
сmetrics
Ptrainable_variables
тnon_trainable_variables
_]
VARIABLE_VALUEblock3_conv1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock3_conv1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

R0
S1
 

 уlayer_regularization_losses
фlayers
Uregularization_losses
V	variables
хmetrics
Wtrainable_variables
цnon_trainable_variables
_]
VARIABLE_VALUEblock3_conv2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock3_conv2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

Y0
Z1
 

 чlayer_regularization_losses
шlayers
\regularization_losses
]	variables
щmetrics
^trainable_variables
ъnon_trainable_variables
_]
VARIABLE_VALUEblock3_conv3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock3_conv3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

`0
a1
 

 ыlayer_regularization_losses
ьlayers
cregularization_losses
d	variables
эmetrics
etrainable_variables
юnon_trainable_variables
 
 
 
 

 яlayer_regularization_losses
№layers
hregularization_losses
i	variables
ёmetrics
jtrainable_variables
ђnon_trainable_variables
_]
VARIABLE_VALUEblock4_conv1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock4_conv1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

l0
m1
 

 ѓlayer_regularization_losses
єlayers
oregularization_losses
p	variables
ѕmetrics
qtrainable_variables
іnon_trainable_variables
_]
VARIABLE_VALUEblock4_conv2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock4_conv2/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

s0
t1
 

 їlayer_regularization_losses
јlayers
vregularization_losses
w	variables
љmetrics
xtrainable_variables
њnon_trainable_variables
_]
VARIABLE_VALUEblock4_conv3/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock4_conv3/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

z0
{1
 

 ћlayer_regularization_losses
ќlayers
}regularization_losses
~	variables
§metrics
trainable_variables
ўnon_trainable_variables
 
 
 
 
Ё
 џlayer_regularization_losses
layers
regularization_losses
	variables
metrics
trainable_variables
non_trainable_variables
`^
VARIABLE_VALUEblock5_conv1/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock5_conv1/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
 
Ё
 layer_regularization_losses
layers
regularization_losses
	variables
metrics
trainable_variables
non_trainable_variables
`^
VARIABLE_VALUEblock5_conv2/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock5_conv2/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
 
Ё
 layer_regularization_losses
layers
regularization_losses
	variables
metrics
trainable_variables
non_trainable_variables
`^
VARIABLE_VALUEblock5_conv3/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock5_conv3/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
 
Ё
 layer_regularization_losses
layers
regularization_losses
	variables
metrics
trainable_variables
non_trainable_variables
 
 
 
 
Ё
 layer_regularization_losses
layers
regularization_losses
	variables
metrics
trainable_variables
non_trainable_variables
 
 
 
 
Ё
 layer_regularization_losses
layers
Ёregularization_losses
Ђ	variables
metrics
Ѓtrainable_variables
non_trainable_variables
WU
VARIABLE_VALUE
fc1/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEfc1/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

Ѕ0
І1

Ѕ0
І1
Ё
 layer_regularization_losses
layers
Јregularization_losses
Љ	variables
metrics
Њtrainable_variables
non_trainable_variables
WU
VARIABLE_VALUE
fc2/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEfc2/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

Ќ0
­1

Ќ0
­1
Ё
 layer_regularization_losses
layers
Џregularization_losses
А	variables
metrics
Бtrainable_variables
non_trainable_variables
_]
VARIABLE_VALUEpredictions/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEpredictions/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

Г0
Д1

Г0
Д1
Ё
 layer_regularization_losses
 layers
Жregularization_losses
З	variables
Ёmetrics
Иtrainable_variables
Ђnon_trainable_variables
QO
VARIABLE_VALUEtraining/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEtraining/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEtraining/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
Ж
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23

Ѓ0
Ь
,0
-1
32
43
?4
@5
F6
G7
R8
S9
Y10
Z11
`12
a13
l14
m15
s16
t17
z18
{19
20
21
22
23
24
25
 
 
 
 
 
 
 
 
 
 
 

,0
-1
 
 
 

30
41
 
 
 
 
 
 
 

?0
@1
 
 
 

F0
G1
 
 
 
 
 
 
 

R0
S1
 
 
 

Y0
Z1
 
 
 

`0
a1
 
 
 
 
 
 
 

l0
m1
 
 
 

s0
t1
 
 
 

z0
{1
 
 
 
 
 
 
 

0
1
 
 
 

0
1
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


Єtotal

Ѕcount
І
_fn_kwargs
Ї_updates
Јregularization_losses
Љ	variables
Њtrainable_variables
Ћ	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

Є0
Ѕ1
 
Ё
 Ќlayer_regularization_losses
­layers
Јregularization_losses
Љ	variables
Ўmetrics
Њtrainable_variables
Џnon_trainable_variables
 
 
 

Є0
Ѕ1

VARIABLE_VALUEtraining/Adam/fc1/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEtraining/Adam/fc1/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEtraining/Adam/fc2/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEtraining/Adam/fc2/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"training/Adam/predictions/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE training/Adam/predictions/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEtraining/Adam/fc1/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEtraining/Adam/fc1/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEtraining/Adam/fc2/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEtraining/Adam/fc2/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"training/Adam/predictions/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE training/Adam/predictions/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 

serving_default_input_1Placeholder*&
shape:џџџџџџџџџрр*
dtype0*1
_output_shapes
:џџџџџџџџџрр
 
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1block1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/bias
fc1/kernelfc1/bias
fc2/kernelfc2/biaspredictions/kernelpredictions/biasConst*/
config_proto

CPU

GPU2 *0J 8*-
Tin&
$2"*'
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-19996*,
f'R%
#__inference_signature_wrapper_19467*
Tout
2
O
saver_filenamePlaceholder*
_output_shapes
: *
shape: *
dtype0
ј
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'block1_conv1/kernel/Read/ReadVariableOp%block1_conv1/bias/Read/ReadVariableOp'block1_conv2/kernel/Read/ReadVariableOp%block1_conv2/bias/Read/ReadVariableOp'block2_conv1/kernel/Read/ReadVariableOp%block2_conv1/bias/Read/ReadVariableOp'block2_conv2/kernel/Read/ReadVariableOp%block2_conv2/bias/Read/ReadVariableOp'block3_conv1/kernel/Read/ReadVariableOp%block3_conv1/bias/Read/ReadVariableOp'block3_conv2/kernel/Read/ReadVariableOp%block3_conv2/bias/Read/ReadVariableOp'block3_conv3/kernel/Read/ReadVariableOp%block3_conv3/bias/Read/ReadVariableOp'block4_conv1/kernel/Read/ReadVariableOp%block4_conv1/bias/Read/ReadVariableOp'block4_conv2/kernel/Read/ReadVariableOp%block4_conv2/bias/Read/ReadVariableOp'block4_conv3/kernel/Read/ReadVariableOp%block4_conv3/bias/Read/ReadVariableOp'block5_conv1/kernel/Read/ReadVariableOp%block5_conv1/bias/Read/ReadVariableOp'block5_conv2/kernel/Read/ReadVariableOp%block5_conv2/bias/Read/ReadVariableOp'block5_conv3/kernel/Read/ReadVariableOp%block5_conv3/bias/Read/ReadVariableOpfc1/kernel/Read/ReadVariableOpfc1/bias/Read/ReadVariableOpfc2/kernel/Read/ReadVariableOpfc2/bias/Read/ReadVariableOp&predictions/kernel/Read/ReadVariableOp$predictions/bias/Read/ReadVariableOp&training/Adam/iter/Read/ReadVariableOp(training/Adam/beta_1/Read/ReadVariableOp(training/Adam/beta_2/Read/ReadVariableOp'training/Adam/decay/Read/ReadVariableOp/training/Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp.training/Adam/fc1/kernel/m/Read/ReadVariableOp,training/Adam/fc1/bias/m/Read/ReadVariableOp.training/Adam/fc2/kernel/m/Read/ReadVariableOp,training/Adam/fc2/bias/m/Read/ReadVariableOp6training/Adam/predictions/kernel/m/Read/ReadVariableOp4training/Adam/predictions/bias/m/Read/ReadVariableOp.training/Adam/fc1/kernel/v/Read/ReadVariableOp,training/Adam/fc1/bias/v/Read/ReadVariableOp.training/Adam/fc2/kernel/v/Read/ReadVariableOp,training/Adam/fc2/bias/v/Read/ReadVariableOp6training/Adam/predictions/kernel/v/Read/ReadVariableOp4training/Adam/predictions/bias/v/Read/ReadVariableOpConst_1*/
config_proto

CPU

GPU2 *0J 8*@
Tin9
725	*
_output_shapes
: *,
_gradient_op_typePartitionedCall-20069*'
f"R 
__inference__traced_save_20068*
Tout
2
ѕ

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/bias
fc1/kernelfc1/bias
fc2/kernelfc2/biaspredictions/kernelpredictions/biastraining/Adam/itertraining/Adam/beta_1training/Adam/beta_2training/Adam/decaytraining/Adam/learning_ratetotalcounttraining/Adam/fc1/kernel/mtraining/Adam/fc1/bias/mtraining/Adam/fc2/kernel/mtraining/Adam/fc2/bias/m"training/Adam/predictions/kernel/m training/Adam/predictions/bias/mtraining/Adam/fc1/kernel/vtraining/Adam/fc1/bias/vtraining/Adam/fc2/kernel/vtraining/Adam/fc2/bias/v"training/Adam/predictions/kernel/v training/Adam/predictions/bias/v*,
_gradient_op_typePartitionedCall-20235**
f%R#
!__inference__traced_restore_20234*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*?
Tin8
624*
_output_shapes
: дЄ

w
A__inference_lambda_layer_call_and_return_conditional_losses_19006

inputs
biasadd_lambda_const
identityd
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_2Const*
valueB"   џџџџ*
dtype0*
_output_shapes
:
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*

begin_mask*
ellipsis_mask*
end_mask*1
_output_shapes
:џџџџџџџџџрр|
BiasAddBiasAddstrided_slice:output:0biasadd_lambda_const*
T0*1
_output_shapes
:џџџџџџџџџррb
IdentityIdentityBiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџрр"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџрр::& "
 
_user_specified_nameinputs: 
 \
ю
__inference__traced_save_20068
file_prefix2
.savev2_block1_conv1_kernel_read_readvariableop0
,savev2_block1_conv1_bias_read_readvariableop2
.savev2_block1_conv2_kernel_read_readvariableop0
,savev2_block1_conv2_bias_read_readvariableop2
.savev2_block2_conv1_kernel_read_readvariableop0
,savev2_block2_conv1_bias_read_readvariableop2
.savev2_block2_conv2_kernel_read_readvariableop0
,savev2_block2_conv2_bias_read_readvariableop2
.savev2_block3_conv1_kernel_read_readvariableop0
,savev2_block3_conv1_bias_read_readvariableop2
.savev2_block3_conv2_kernel_read_readvariableop0
,savev2_block3_conv2_bias_read_readvariableop2
.savev2_block3_conv3_kernel_read_readvariableop0
,savev2_block3_conv3_bias_read_readvariableop2
.savev2_block4_conv1_kernel_read_readvariableop0
,savev2_block4_conv1_bias_read_readvariableop2
.savev2_block4_conv2_kernel_read_readvariableop0
,savev2_block4_conv2_bias_read_readvariableop2
.savev2_block4_conv3_kernel_read_readvariableop0
,savev2_block4_conv3_bias_read_readvariableop2
.savev2_block5_conv1_kernel_read_readvariableop0
,savev2_block5_conv1_bias_read_readvariableop2
.savev2_block5_conv2_kernel_read_readvariableop0
,savev2_block5_conv2_bias_read_readvariableop2
.savev2_block5_conv3_kernel_read_readvariableop0
,savev2_block5_conv3_bias_read_readvariableop)
%savev2_fc1_kernel_read_readvariableop'
#savev2_fc1_bias_read_readvariableop)
%savev2_fc2_kernel_read_readvariableop'
#savev2_fc2_bias_read_readvariableop1
-savev2_predictions_kernel_read_readvariableop/
+savev2_predictions_bias_read_readvariableop1
-savev2_training_adam_iter_read_readvariableop	3
/savev2_training_adam_beta_1_read_readvariableop3
/savev2_training_adam_beta_2_read_readvariableop2
.savev2_training_adam_decay_read_readvariableop:
6savev2_training_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop9
5savev2_training_adam_fc1_kernel_m_read_readvariableop7
3savev2_training_adam_fc1_bias_m_read_readvariableop9
5savev2_training_adam_fc2_kernel_m_read_readvariableop7
3savev2_training_adam_fc2_bias_m_read_readvariableopA
=savev2_training_adam_predictions_kernel_m_read_readvariableop?
;savev2_training_adam_predictions_bias_m_read_readvariableop9
5savev2_training_adam_fc1_kernel_v_read_readvariableop7
3savev2_training_adam_fc1_bias_v_read_readvariableop9
5savev2_training_adam_fc2_kernel_v_read_readvariableop7
3savev2_training_adam_fc2_bias_v_read_readvariableopA
=savev2_training_adam_predictions_kernel_v_read_readvariableop?
;savev2_training_adam_predictions_bias_v_read_readvariableop
savev2_1_const_1

identity_1ЂMergeV2CheckpointsЂSaveV2ЂSaveV2_1
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_14e4e3cc0eba49fb9e9d5e6dfcce953c/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
value	B : *
dtype0
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*Т
valueИBЕ3B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:3г
SaveV2/shape_and_slicesConst"/device:CPU:0*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:3ѓ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_block1_conv1_kernel_read_readvariableop,savev2_block1_conv1_bias_read_readvariableop.savev2_block1_conv2_kernel_read_readvariableop,savev2_block1_conv2_bias_read_readvariableop.savev2_block2_conv1_kernel_read_readvariableop,savev2_block2_conv1_bias_read_readvariableop.savev2_block2_conv2_kernel_read_readvariableop,savev2_block2_conv2_bias_read_readvariableop.savev2_block3_conv1_kernel_read_readvariableop,savev2_block3_conv1_bias_read_readvariableop.savev2_block3_conv2_kernel_read_readvariableop,savev2_block3_conv2_bias_read_readvariableop.savev2_block3_conv3_kernel_read_readvariableop,savev2_block3_conv3_bias_read_readvariableop.savev2_block4_conv1_kernel_read_readvariableop,savev2_block4_conv1_bias_read_readvariableop.savev2_block4_conv2_kernel_read_readvariableop,savev2_block4_conv2_bias_read_readvariableop.savev2_block4_conv3_kernel_read_readvariableop,savev2_block4_conv3_bias_read_readvariableop.savev2_block5_conv1_kernel_read_readvariableop,savev2_block5_conv1_bias_read_readvariableop.savev2_block5_conv2_kernel_read_readvariableop,savev2_block5_conv2_bias_read_readvariableop.savev2_block5_conv3_kernel_read_readvariableop,savev2_block5_conv3_bias_read_readvariableop%savev2_fc1_kernel_read_readvariableop#savev2_fc1_bias_read_readvariableop%savev2_fc2_kernel_read_readvariableop#savev2_fc2_bias_read_readvariableop-savev2_predictions_kernel_read_readvariableop+savev2_predictions_bias_read_readvariableop-savev2_training_adam_iter_read_readvariableop/savev2_training_adam_beta_1_read_readvariableop/savev2_training_adam_beta_2_read_readvariableop.savev2_training_adam_decay_read_readvariableop6savev2_training_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop5savev2_training_adam_fc1_kernel_m_read_readvariableop3savev2_training_adam_fc1_bias_m_read_readvariableop5savev2_training_adam_fc2_kernel_m_read_readvariableop3savev2_training_adam_fc2_bias_m_read_readvariableop=savev2_training_adam_predictions_kernel_m_read_readvariableop;savev2_training_adam_predictions_bias_m_read_readvariableop5savev2_training_adam_fc1_kernel_v_read_readvariableop3savev2_training_adam_fc1_bias_v_read_readvariableop5savev2_training_adam_fc2_kernel_v_read_readvariableop3savev2_training_adam_fc2_bias_v_read_readvariableop=savev2_training_adam_predictions_kernel_v_read_readvariableop;savev2_training_adam_predictions_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *A
dtypes7
523	h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:Х
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const_1^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 Й
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0*
_input_shapes
: :@:@:@@:@:@::::::::::::::::::::::
::	:::: : : : : : : :
::	::::
::	:::: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 :4 
Ї
G
+__inference_block4_pool_layer_call_fn_18868

inputs
identityУ
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-18865*O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_18859*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:& "
 
_user_specified_nameinputs
ўo
р
@__inference_model_layer_call_and_return_conditional_losses_19156
input_1'
#lambda_partitionedcall_lambda_const/
+block1_conv1_statefulpartitionedcall_args_1/
+block1_conv1_statefulpartitionedcall_args_2/
+block1_conv2_statefulpartitionedcall_args_1/
+block1_conv2_statefulpartitionedcall_args_2/
+block2_conv1_statefulpartitionedcall_args_1/
+block2_conv1_statefulpartitionedcall_args_2/
+block2_conv2_statefulpartitionedcall_args_1/
+block2_conv2_statefulpartitionedcall_args_2/
+block3_conv1_statefulpartitionedcall_args_1/
+block3_conv1_statefulpartitionedcall_args_2/
+block3_conv2_statefulpartitionedcall_args_1/
+block3_conv2_statefulpartitionedcall_args_2/
+block3_conv3_statefulpartitionedcall_args_1/
+block3_conv3_statefulpartitionedcall_args_2/
+block4_conv1_statefulpartitionedcall_args_1/
+block4_conv1_statefulpartitionedcall_args_2/
+block4_conv2_statefulpartitionedcall_args_1/
+block4_conv2_statefulpartitionedcall_args_2/
+block4_conv3_statefulpartitionedcall_args_1/
+block4_conv3_statefulpartitionedcall_args_2/
+block5_conv1_statefulpartitionedcall_args_1/
+block5_conv1_statefulpartitionedcall_args_2/
+block5_conv2_statefulpartitionedcall_args_1/
+block5_conv2_statefulpartitionedcall_args_2/
+block5_conv3_statefulpartitionedcall_args_1/
+block5_conv3_statefulpartitionedcall_args_2&
"fc1_statefulpartitionedcall_args_1&
"fc1_statefulpartitionedcall_args_2&
"fc2_statefulpartitionedcall_args_1&
"fc2_statefulpartitionedcall_args_2.
*predictions_statefulpartitionedcall_args_1.
*predictions_statefulpartitionedcall_args_2
identityЂ$block1_conv1/StatefulPartitionedCallЂ$block1_conv2/StatefulPartitionedCallЂ$block2_conv1/StatefulPartitionedCallЂ$block2_conv2/StatefulPartitionedCallЂ$block3_conv1/StatefulPartitionedCallЂ$block3_conv2/StatefulPartitionedCallЂ$block3_conv3/StatefulPartitionedCallЂ$block4_conv1/StatefulPartitionedCallЂ$block4_conv2/StatefulPartitionedCallЂ$block4_conv3/StatefulPartitionedCallЂ$block5_conv1/StatefulPartitionedCallЂ$block5_conv2/StatefulPartitionedCallЂ$block5_conv3/StatefulPartitionedCallЂfc1/StatefulPartitionedCallЂfc2/StatefulPartitionedCallЂ#predictions/StatefulPartitionedCallг
lambda/PartitionedCallPartitionedCallinput_1#lambda_partitionedcall_lambda_const*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*1
_output_shapes
:џџџџџџџџџрр*,
_gradient_op_typePartitionedCall-19010*J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_18994Н
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0+block1_conv1_statefulpartitionedcall_args_1+block1_conv1_statefulpartitionedcall_args_2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*1
_output_shapes
:џџџџџџџџџрр@*,
_gradient_op_typePartitionedCall-18570*P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_18564*
Tout
2Ы
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0+block1_conv2_statefulpartitionedcall_args_1+block1_conv2_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-18595*P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_18589*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*1
_output_shapes
:џџџџџџџџџрр@*
Tin
2л
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџpp@*,
_gradient_op_typePartitionedCall-18614*O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_18608*
Tout
2С
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0+block2_conv1_statefulpartitionedcall_args_1+block2_conv1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-18637*P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_18631*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџppЪ
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0+block2_conv2_statefulpartitionedcall_args_1+block2_conv2_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-18662*P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_18656*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*0
_output_shapes
:џџџџџџџџџpp*
Tin
2м
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџ88*,
_gradient_op_typePartitionedCall-18681*O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_18675*
Tout
2С
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0+block3_conv1_statefulpartitionedcall_args_1+block3_conv1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-18704*P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_18698*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџ88Ъ
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0+block3_conv2_statefulpartitionedcall_args_1+block3_conv2_statefulpartitionedcall_args_2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџ88*,
_gradient_op_typePartitionedCall-18729*P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_18723*
Tout
2Ъ
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0+block3_conv3_statefulpartitionedcall_args_1+block3_conv3_statefulpartitionedcall_args_2*/
config_proto

CPU

GPU2 *0J 8*0
_output_shapes
:џџџџџџџџџ88*
Tin
2*,
_gradient_op_typePartitionedCall-18754*P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_18748*
Tout
2м
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-18773*O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_18767С
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0+block4_conv1_statefulpartitionedcall_args_1+block4_conv1_statefulpartitionedcall_args_2*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-18796*P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_18790Ъ
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0+block4_conv2_statefulpartitionedcall_args_1+block4_conv2_statefulpartitionedcall_args_2*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-18821*P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_18815Ъ
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0+block4_conv3_statefulpartitionedcall_args_1+block4_conv3_statefulpartitionedcall_args_2*0
_output_shapes
:џџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-18846*P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_18840*
Tout
2*/
config_proto

CPU

GPU2 *0J 8м
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-18865*O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_18859*
Tout
2С
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0+block5_conv1_statefulpartitionedcall_args_1+block5_conv1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-18888*P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_18882*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџЪ
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0+block5_conv2_statefulpartitionedcall_args_1+block5_conv2_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-18913*P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_18907*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџЪ
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0+block5_conv3_statefulpartitionedcall_args_1+block5_conv3_statefulpartitionedcall_args_2*
Tin
2*0
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-18938*P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_18932*
Tout
2*/
config_proto

CPU

GPU2 *0J 8м
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-18957*O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_18951*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџх
(global_average_pooling2d/PartitionedCallPartitionedCall$block5_pool/PartitionedCall:output:0*(
_output_shapes
:џџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-18975*\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_18969*
Tout
2*/
config_proto

CPU

GPU2 *0J 8Ђ
fc1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0"fc1_statefulpartitionedcall_args_1"fc1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-19088*G
fBR@
>__inference_fc1_layer_call_and_return_conditional_losses_19082*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*(
_output_shapes
:џџџџџџџџџ*
Tin
2
fc2/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0"fc2_statefulpartitionedcall_args_1"fc2_statefulpartitionedcall_args_2*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*'
_output_shapes
:џџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-19116*G
fBR@
>__inference_fc2_layer_call_and_return_conditional_losses_19110Д
#predictions/StatefulPartitionedCallStatefulPartitionedCall$fc2/StatefulPartitionedCall:output:0*predictions_statefulpartitionedcall_args_1*predictions_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-19144*O
fJRH
F__inference_predictions_layer_call_and_return_conditional_losses_19138*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*'
_output_shapes
:џџџџџџџџџ*
Tin
2б
IdentityIdentity,predictions/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall$^predictions/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*И
_input_shapesІ
Ѓ:џџџџџџџџџрр:::::::::::::::::::::::::::::::::2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2J
#predictions/StatefulPartitionedCall#predictions/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! 
ћo
п
@__inference_model_layer_call_and_return_conditional_losses_19281

inputs'
#lambda_partitionedcall_lambda_const/
+block1_conv1_statefulpartitionedcall_args_1/
+block1_conv1_statefulpartitionedcall_args_2/
+block1_conv2_statefulpartitionedcall_args_1/
+block1_conv2_statefulpartitionedcall_args_2/
+block2_conv1_statefulpartitionedcall_args_1/
+block2_conv1_statefulpartitionedcall_args_2/
+block2_conv2_statefulpartitionedcall_args_1/
+block2_conv2_statefulpartitionedcall_args_2/
+block3_conv1_statefulpartitionedcall_args_1/
+block3_conv1_statefulpartitionedcall_args_2/
+block3_conv2_statefulpartitionedcall_args_1/
+block3_conv2_statefulpartitionedcall_args_2/
+block3_conv3_statefulpartitionedcall_args_1/
+block3_conv3_statefulpartitionedcall_args_2/
+block4_conv1_statefulpartitionedcall_args_1/
+block4_conv1_statefulpartitionedcall_args_2/
+block4_conv2_statefulpartitionedcall_args_1/
+block4_conv2_statefulpartitionedcall_args_2/
+block4_conv3_statefulpartitionedcall_args_1/
+block4_conv3_statefulpartitionedcall_args_2/
+block5_conv1_statefulpartitionedcall_args_1/
+block5_conv1_statefulpartitionedcall_args_2/
+block5_conv2_statefulpartitionedcall_args_1/
+block5_conv2_statefulpartitionedcall_args_2/
+block5_conv3_statefulpartitionedcall_args_1/
+block5_conv3_statefulpartitionedcall_args_2&
"fc1_statefulpartitionedcall_args_1&
"fc1_statefulpartitionedcall_args_2&
"fc2_statefulpartitionedcall_args_1&
"fc2_statefulpartitionedcall_args_2.
*predictions_statefulpartitionedcall_args_1.
*predictions_statefulpartitionedcall_args_2
identityЂ$block1_conv1/StatefulPartitionedCallЂ$block1_conv2/StatefulPartitionedCallЂ$block2_conv1/StatefulPartitionedCallЂ$block2_conv2/StatefulPartitionedCallЂ$block3_conv1/StatefulPartitionedCallЂ$block3_conv2/StatefulPartitionedCallЂ$block3_conv3/StatefulPartitionedCallЂ$block4_conv1/StatefulPartitionedCallЂ$block4_conv2/StatefulPartitionedCallЂ$block4_conv3/StatefulPartitionedCallЂ$block5_conv1/StatefulPartitionedCallЂ$block5_conv2/StatefulPartitionedCallЂ$block5_conv3/StatefulPartitionedCallЂfc1/StatefulPartitionedCallЂfc2/StatefulPartitionedCallЂ#predictions/StatefulPartitionedCallв
lambda/PartitionedCallPartitionedCallinputs#lambda_partitionedcall_lambda_const*J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_18994*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*1
_output_shapes
:џџџџџџџџџрр*
Tin
2*,
_gradient_op_typePartitionedCall-19010Н
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0+block1_conv1_statefulpartitionedcall_args_1+block1_conv1_statefulpartitionedcall_args_2*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*1
_output_shapes
:џџџџџџџџџрр@*,
_gradient_op_typePartitionedCall-18570*P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_18564Ы
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0+block1_conv2_statefulpartitionedcall_args_1+block1_conv2_statefulpartitionedcall_args_2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*1
_output_shapes
:џџџџџџџџџрр@*,
_gradient_op_typePartitionedCall-18595*P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_18589*
Tout
2л
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџpp@*,
_gradient_op_typePartitionedCall-18614*O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_18608*
Tout
2С
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0+block2_conv1_statefulpartitionedcall_args_1+block2_conv1_statefulpartitionedcall_args_2*0
_output_shapes
:џџџџџџџџџpp*
Tin
2*,
_gradient_op_typePartitionedCall-18637*P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_18631*
Tout
2*/
config_proto

CPU

GPU2 *0J 8Ъ
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0+block2_conv2_statefulpartitionedcall_args_1+block2_conv2_statefulpartitionedcall_args_2*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџpp*,
_gradient_op_typePartitionedCall-18662*P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_18656м
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-18681*O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_18675*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*0
_output_shapes
:џџџџџџџџџ88*
Tin
2С
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0+block3_conv1_statefulpartitionedcall_args_1+block3_conv1_statefulpartitionedcall_args_2*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*0
_output_shapes
:џџџџџџџџџ88*
Tin
2*,
_gradient_op_typePartitionedCall-18704*P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_18698Ъ
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0+block3_conv2_statefulpartitionedcall_args_1+block3_conv2_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-18729*P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_18723*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџ88Ъ
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0+block3_conv3_statefulpartitionedcall_args_1+block3_conv3_statefulpartitionedcall_args_2*/
config_proto

CPU

GPU2 *0J 8*0
_output_shapes
:џџџџџџџџџ88*
Tin
2*,
_gradient_op_typePartitionedCall-18754*P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_18748*
Tout
2м
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*0
_output_shapes
:џџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-18773*O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_18767С
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0+block4_conv1_statefulpartitionedcall_args_1+block4_conv1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-18796*P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_18790*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџЪ
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0+block4_conv2_statefulpartitionedcall_args_1+block4_conv2_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-18821*P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_18815*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџЪ
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0+block4_conv3_statefulpartitionedcall_args_1+block4_conv3_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-18846*P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_18840*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџм
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-18865*O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_18859*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*0
_output_shapes
:џџџџџџџџџ*
Tin
2С
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0+block5_conv1_statefulpartitionedcall_args_1+block5_conv1_statefulpartitionedcall_args_2*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-18888*P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_18882Ъ
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0+block5_conv2_statefulpartitionedcall_args_1+block5_conv2_statefulpartitionedcall_args_2*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-18913*P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_18907Ъ
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0+block5_conv3_statefulpartitionedcall_args_1+block5_conv3_statefulpartitionedcall_args_2*/
config_proto

CPU

GPU2 *0J 8*0
_output_shapes
:џџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-18938*P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_18932*
Tout
2м
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-18957*O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_18951*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*0
_output_shapes
:џџџџџџџџџ*
Tin
2х
(global_average_pooling2d/PartitionedCallPartitionedCall$block5_pool/PartitionedCall:output:0*/
config_proto

CPU

GPU2 *0J 8*(
_output_shapes
:џџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-18975*\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_18969*
Tout
2Ђ
fc1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0"fc1_statefulpartitionedcall_args_1"fc1_statefulpartitionedcall_args_2*/
config_proto

CPU

GPU2 *0J 8*(
_output_shapes
:џџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-19088*G
fBR@
>__inference_fc1_layer_call_and_return_conditional_losses_19082*
Tout
2
fc2/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0"fc2_statefulpartitionedcall_args_1"fc2_statefulpartitionedcall_args_2*
Tin
2*'
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-19116*G
fBR@
>__inference_fc2_layer_call_and_return_conditional_losses_19110*
Tout
2*/
config_proto

CPU

GPU2 *0J 8Д
#predictions/StatefulPartitionedCallStatefulPartitionedCall$fc2/StatefulPartitionedCall:output:0*predictions_statefulpartitionedcall_args_1*predictions_statefulpartitionedcall_args_2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*'
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-19144*O
fJRH
F__inference_predictions_layer_call_and_return_conditional_losses_19138*
Tout
2б
IdentityIdentity,predictions/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall$^predictions/StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*И
_input_shapesІ
Ѓ:џџџџџџџџџрр:::::::::::::::::::::::::::::::::2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2J
#predictions/StatefulPartitionedCall#predictions/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall: : : :  :! :& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : 
Ї
G
+__inference_block1_pool_layer_call_fn_18617

inputs
identityУ
PartitionedCallPartitionedCallinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-18614*O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_18608*
Tout
2*/
config_proto

CPU

GPU2 *0J 8
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:& "
 
_user_specified_nameinputs
А
 

%__inference_model_layer_call_fn_19803

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33(
$statefulpartitionedcall_lambda_const
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs$statefulpartitionedcall_lambda_conststatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_19382*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*'
_output_shapes
:џџџџџџџџџ*-
Tin&
$2"*,
_gradient_op_typePartitionedCall-19383
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*И
_input_shapesІ
Ѓ:џџџџџџџџџрр:::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! 

р
G__inference_block4_conv2_layer_call_and_return_conditional_losses_18815

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:Ќ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0*
strides
*
paddingSAMEЁ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџІ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
б
Є
#__inference_fc2_layer_call_fn_19871

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*'
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-19116*G
fBR@
>__inference_fc2_layer_call_and_return_conditional_losses_19110*
Tout
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 

b
F__inference_block1_pool_layer_call_and_return_conditional_losses_18608

inputs
identityЂ
MaxPoolMaxPoolinputs*
strides
*
ksize
*
paddingVALID*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ{
IdentityIdentityMaxPool:output:0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:& "
 
_user_specified_nameinputs
Г
­
,__inference_block3_conv1_layer_call_fn_18709

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
config_proto

CPU

GPU2 *0J 8*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-18704*P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_18698*
Tout
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
Г
­
,__inference_block3_conv3_layer_call_fn_18759

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-18754*P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_18748*
Tout
2*/
config_proto

CPU

GPU2 *0J 8
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
Ї
G
+__inference_block5_pool_layer_call_fn_18960

inputs
identityУ
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-18957*O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_18951*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
IdentityIdentityPartitionedCall:output:0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:& "
 
_user_specified_nameinputs

w
A__inference_lambda_layer_call_and_return_conditional_losses_19813

inputs
biasadd_lambda_const
identityd
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0f
strided_slice/stack_2Const*
valueB"   џџџџ*
dtype0*
_output_shapes
:
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*

begin_mask*
ellipsis_mask*
end_mask*1
_output_shapes
:џџџџџџџџџрр*
T0*
Index0|
BiasAddBiasAddstrided_slice:output:0biasadd_lambda_const*
T0*1
_output_shapes
:џџџџџџџџџррb
IdentityIdentityBiasAdd:output:0*1
_output_shapes
:џџџџџџџџџрр*
T0"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџрр::& "
 
_user_specified_nameinputs: 
А
­
,__inference_block1_conv2_layer_call_fn_18600

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*,
_gradient_op_typePartitionedCall-18595*P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_18589
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 

р
G__inference_block4_conv3_layer_call_and_return_conditional_losses_18840

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:Ќ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0Ё
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0k
ReluReluBiasAdd:output:0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0І
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
з	
п
F__inference_predictions_layer_call_and_return_conditional_losses_19138

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ*
T0 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 

р
G__inference_block2_conv2_layer_call_and_return_conditional_losses_18656

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*(
_output_shapes
:*
dtype0Ќ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџЁ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџk
ReluReluBiasAdd:output:0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0І
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
в	
з
>__inference_fc1_layer_call_and_return_conditional_losses_19082

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЄ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЁ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
ReluReluBiasAdd:output:0*(
_output_shapes
:џџџџџџџџџ*
T0
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 

р
G__inference_block3_conv1_layer_call_and_return_conditional_losses_18698

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*(
_output_shapes
:*
dtype0Ќ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџЁ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0k
ReluReluBiasAdd:output:0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0І
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
Г
­
,__inference_block5_conv2_layer_call_fn_18918

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-18913*P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_18907*
Tout
2*/
config_proto

CPU

GPU2 *0J 8
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 

р
G__inference_block3_conv2_layer_call_and_return_conditional_losses_18723

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*(
_output_shapes
:*
dtype0Ќ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџЁ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџk
ReluReluBiasAdd:output:0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0І
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 

d
&__inference_lambda_layer_call_fn_19829

inputs 
partitionedcall_lambda_const
identityФ
PartitionedCallPartitionedCallinputspartitionedcall_lambda_const*
Tin
2*1
_output_shapes
:џџџџџџџџџрр*,
_gradient_op_typePartitionedCall-19010*J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_18994*
Tout
2*/
config_proto

CPU

GPU2 *0J 8j
IdentityIdentityPartitionedCall:output:0*1
_output_shapes
:џџџџџџџџџрр*
T0"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџрр::& "
 
_user_specified_nameinputs: 
Г
­
,__inference_block5_conv1_layer_call_fn_18893

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-18888*P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_18882*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
Г
­
,__inference_block5_conv3_layer_call_fn_18943

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-18938*P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_18932*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
Ь	
з
>__inference_fc2_layer_call_and_return_conditional_losses_19110

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЃ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
А

@__inference_model_layer_call_and_return_conditional_losses_19727

inputs
lambda_biasadd_lambda_const/
+block1_conv1_conv2d_readvariableop_resource0
,block1_conv1_biasadd_readvariableop_resource/
+block1_conv2_conv2d_readvariableop_resource0
,block1_conv2_biasadd_readvariableop_resource/
+block2_conv1_conv2d_readvariableop_resource0
,block2_conv1_biasadd_readvariableop_resource/
+block2_conv2_conv2d_readvariableop_resource0
,block2_conv2_biasadd_readvariableop_resource/
+block3_conv1_conv2d_readvariableop_resource0
,block3_conv1_biasadd_readvariableop_resource/
+block3_conv2_conv2d_readvariableop_resource0
,block3_conv2_biasadd_readvariableop_resource/
+block3_conv3_conv2d_readvariableop_resource0
,block3_conv3_biasadd_readvariableop_resource/
+block4_conv1_conv2d_readvariableop_resource0
,block4_conv1_biasadd_readvariableop_resource/
+block4_conv2_conv2d_readvariableop_resource0
,block4_conv2_biasadd_readvariableop_resource/
+block4_conv3_conv2d_readvariableop_resource0
,block4_conv3_biasadd_readvariableop_resource/
+block5_conv1_conv2d_readvariableop_resource0
,block5_conv1_biasadd_readvariableop_resource/
+block5_conv2_conv2d_readvariableop_resource0
,block5_conv2_biasadd_readvariableop_resource/
+block5_conv3_conv2d_readvariableop_resource0
,block5_conv3_biasadd_readvariableop_resource&
"fc1_matmul_readvariableop_resource'
#fc1_biasadd_readvariableop_resource&
"fc2_matmul_readvariableop_resource'
#fc2_biasadd_readvariableop_resource.
*predictions_matmul_readvariableop_resource/
+predictions_biasadd_readvariableop_resource
identityЂ#block1_conv1/BiasAdd/ReadVariableOpЂ"block1_conv1/Conv2D/ReadVariableOpЂ#block1_conv2/BiasAdd/ReadVariableOpЂ"block1_conv2/Conv2D/ReadVariableOpЂ#block2_conv1/BiasAdd/ReadVariableOpЂ"block2_conv1/Conv2D/ReadVariableOpЂ#block2_conv2/BiasAdd/ReadVariableOpЂ"block2_conv2/Conv2D/ReadVariableOpЂ#block3_conv1/BiasAdd/ReadVariableOpЂ"block3_conv1/Conv2D/ReadVariableOpЂ#block3_conv2/BiasAdd/ReadVariableOpЂ"block3_conv2/Conv2D/ReadVariableOpЂ#block3_conv3/BiasAdd/ReadVariableOpЂ"block3_conv3/Conv2D/ReadVariableOpЂ#block4_conv1/BiasAdd/ReadVariableOpЂ"block4_conv1/Conv2D/ReadVariableOpЂ#block4_conv2/BiasAdd/ReadVariableOpЂ"block4_conv2/Conv2D/ReadVariableOpЂ#block4_conv3/BiasAdd/ReadVariableOpЂ"block4_conv3/Conv2D/ReadVariableOpЂ#block5_conv1/BiasAdd/ReadVariableOpЂ"block5_conv1/Conv2D/ReadVariableOpЂ#block5_conv2/BiasAdd/ReadVariableOpЂ"block5_conv2/Conv2D/ReadVariableOpЂ#block5_conv3/BiasAdd/ReadVariableOpЂ"block5_conv3/Conv2D/ReadVariableOpЂfc1/BiasAdd/ReadVariableOpЂfc1/MatMul/ReadVariableOpЂfc2/BiasAdd/ReadVariableOpЂfc2/MatMul/ReadVariableOpЂ"predictions/BiasAdd/ReadVariableOpЂ!predictions/MatMul/ReadVariableOpk
lambda/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:m
lambda/strided_slice/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:m
lambda/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"   џџџџ
lambda/strided_sliceStridedSliceinputs#lambda/strided_slice/stack:output:0%lambda/strided_slice/stack_1:output:0%lambda/strided_slice/stack_2:output:0*
Index0*
T0*
ellipsis_mask*

begin_mask*
end_mask*1
_output_shapes
:џџџџџџџџџрр
lambda/BiasAddBiasAddlambda/strided_slice:output:0lambda_biasadd_lambda_const*
T0*1
_output_shapes
:џџџџџџџџџррФ
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@Ц
block1_conv1/Conv2DConv2Dlambda/BiasAdd:output:0*block1_conv1/Conv2D/ReadVariableOp:value:0*
paddingSAME*1
_output_shapes
:џџџџџџџџџрр@*
T0*
strides
К
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@І
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџрр@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџрр@Ф
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:@@*
dtype0Ю
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*1
_output_shapes
:џџџџџџџџџрр@*
T0*
strides
*
paddingSAMEК
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@І
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџрр@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџрр@Ќ
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:џџџџџџџџџpp@Х
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*'
_output_shapes
:@Ъ
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:џџџџџџџџџpp*
T0Л
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:Ѕ
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџpps
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџppЦ
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:Э
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:џџџџџџџџџpp*
T0*
strides
Л
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:Ѕ
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџpp*
T0s
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџpp­
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*
ksize
*
paddingVALID*0
_output_shapes
:џџџџџџџџџ88*
strides
Ц
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:Ъ
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:џџџџџџџџџ88*
T0*
strides
Л
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:Ѕ
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ88s
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*0
_output_shapes
:џџџџџџџџџ88*
T0Ц
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*(
_output_shapes
:*
dtype0Э
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:џџџџџџџџџ88*
T0*
strides
Л
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:Ѕ
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ88s
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ88Ц
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:Э
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:џџџџџџџџџ88*
T0*
strides
Л
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:Ѕ
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ88s
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*0
_output_shapes
:џџџџџџџџџ88*
T0­
block3_pool/MaxPoolMaxPoolblock3_conv3/Relu:activations:0*
ksize
*
paddingVALID*0
_output_shapes
:џџџџџџџџџ*
strides
Ц
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:Ъ
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџ*
T0*
strides
*
paddingSAMEЛ
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:*
dtype0Ѕ
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџs
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЦ
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:Э
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџ*
T0*
strides
*
paddingSAMEЛ
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:Ѕ
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџs
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*0
_output_shapes
:џџџџџџџџџ*
T0Ц
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:Э
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџ*
T0*
strides
*
paddingSAMEЛ
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:Ѕ
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџ*
T0s
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*0
_output_shapes
:џџџџџџџџџ*
T0­
block4_pool/MaxPoolMaxPoolblock4_conv3/Relu:activations:0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:џџџџџџџџџЦ
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:Ъ
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:џџџџџџџџџ*
T0*
strides
Л
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:Ѕ
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџs
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*0
_output_shapes
:џџџџџџџџџ*
T0Ц
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:Э
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџ*
T0*
strides
*
paddingSAMEЛ
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:Ѕ
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџ*
T0s
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЦ
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:Э
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџ*
T0*
strides
*
paddingSAMEЛ
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:Ѕ
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџ*
T0s
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ­
block5_pool/MaxPoolMaxPoolblock5_conv3/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
strides
*
ksize
*
paddingVALID
/global_average_pooling2d/Mean/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:А
global_average_pooling2d/MeanMeanblock5_pool/MaxPool:output:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
fc1/MatMul/ReadVariableOpReadVariableOp"fc1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:


fc1/MatMulMatMul&global_average_pooling2d/Mean:output:0!fc1/MatMul/ReadVariableOp:value:0*(
_output_shapes
:џџџџџџџџџ*
T0Љ
fc1/BiasAdd/ReadVariableOpReadVariableOp#fc1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:
fc1/BiasAddBiasAddfc1/MatMul:product:0"fc1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџY
fc1/ReluRelufc1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџЋ
fc2/MatMul/ReadVariableOpReadVariableOp"fc2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	

fc2/MatMulMatMulfc1/Relu:activations:0!fc2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЈ
fc2/BiasAdd/ReadVariableOpReadVariableOp#fc2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:
fc2/BiasAddBiasAddfc2/MatMul:product:0"fc2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџX
fc2/ReluRelufc2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџК
!predictions/MatMul/ReadVariableOpReadVariableOp*predictions_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes

:*
dtype0
predictions/MatMulMatMulfc2/Relu:activations:0)predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџИ
"predictions/BiasAdd/ReadVariableOpReadVariableOp+predictions_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:
predictions/BiasAddBiasAddpredictions/MatMul:product:0*predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџn
predictions/SoftmaxSoftmaxpredictions/BiasAdd:output:0*'
_output_shapes
:џџџџџџџџџ*
T0я	
IdentityIdentitypredictions/Softmax:softmax:0$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp^fc1/BiasAdd/ReadVariableOp^fc1/MatMul/ReadVariableOp^fc2/BiasAdd/ReadVariableOp^fc2/MatMul/ReadVariableOp#^predictions/BiasAdd/ReadVariableOp"^predictions/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*И
_input_shapesІ
Ѓ:џџџџџџџџџрр:::::::::::::::::::::::::::::::::2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp28
fc2/BiasAdd/ReadVariableOpfc2/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2F
!predictions/MatMul/ReadVariableOp!predictions/MatMul/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp28
fc1/BiasAdd/ReadVariableOpfc1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp26
fc1/MatMul/ReadVariableOpfc1/MatMul/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp26
fc2/MatMul/ReadVariableOpfc2/MatMul/ReadVariableOp2H
"predictions/BiasAdd/ReadVariableOp"predictions/BiasAdd/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp: : : : : : : : : : :  :! :& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : 
г
Є
#__inference_fc1_layer_call_fn_19853

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*G
fBR@
>__inference_fc1_layer_call_and_return_conditional_losses_19082*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*(
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-19088
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 

р
G__inference_block1_conv1_layer_call_and_return_conditional_losses_18564

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЊ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
T0 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@j
ReluReluBiasAdd:output:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
T0Ѕ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 

b
F__inference_block3_pool_layer_call_and_return_conditional_losses_18767

inputs
identityЂ
MaxPoolMaxPoolinputs*
strides
*
ksize
*
paddingVALID*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:& "
 
_user_specified_nameinputs
Ї
G
+__inference_block3_pool_layer_call_fn_18776

inputs
identityУ
PartitionedCallPartitionedCallinputs*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-18773*O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_18767
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:& "
 
_user_specified_nameinputs
ћo
п
@__inference_model_layer_call_and_return_conditional_losses_19382

inputs'
#lambda_partitionedcall_lambda_const/
+block1_conv1_statefulpartitionedcall_args_1/
+block1_conv1_statefulpartitionedcall_args_2/
+block1_conv2_statefulpartitionedcall_args_1/
+block1_conv2_statefulpartitionedcall_args_2/
+block2_conv1_statefulpartitionedcall_args_1/
+block2_conv1_statefulpartitionedcall_args_2/
+block2_conv2_statefulpartitionedcall_args_1/
+block2_conv2_statefulpartitionedcall_args_2/
+block3_conv1_statefulpartitionedcall_args_1/
+block3_conv1_statefulpartitionedcall_args_2/
+block3_conv2_statefulpartitionedcall_args_1/
+block3_conv2_statefulpartitionedcall_args_2/
+block3_conv3_statefulpartitionedcall_args_1/
+block3_conv3_statefulpartitionedcall_args_2/
+block4_conv1_statefulpartitionedcall_args_1/
+block4_conv1_statefulpartitionedcall_args_2/
+block4_conv2_statefulpartitionedcall_args_1/
+block4_conv2_statefulpartitionedcall_args_2/
+block4_conv3_statefulpartitionedcall_args_1/
+block4_conv3_statefulpartitionedcall_args_2/
+block5_conv1_statefulpartitionedcall_args_1/
+block5_conv1_statefulpartitionedcall_args_2/
+block5_conv2_statefulpartitionedcall_args_1/
+block5_conv2_statefulpartitionedcall_args_2/
+block5_conv3_statefulpartitionedcall_args_1/
+block5_conv3_statefulpartitionedcall_args_2&
"fc1_statefulpartitionedcall_args_1&
"fc1_statefulpartitionedcall_args_2&
"fc2_statefulpartitionedcall_args_1&
"fc2_statefulpartitionedcall_args_2.
*predictions_statefulpartitionedcall_args_1.
*predictions_statefulpartitionedcall_args_2
identityЂ$block1_conv1/StatefulPartitionedCallЂ$block1_conv2/StatefulPartitionedCallЂ$block2_conv1/StatefulPartitionedCallЂ$block2_conv2/StatefulPartitionedCallЂ$block3_conv1/StatefulPartitionedCallЂ$block3_conv2/StatefulPartitionedCallЂ$block3_conv3/StatefulPartitionedCallЂ$block4_conv1/StatefulPartitionedCallЂ$block4_conv2/StatefulPartitionedCallЂ$block4_conv3/StatefulPartitionedCallЂ$block5_conv1/StatefulPartitionedCallЂ$block5_conv2/StatefulPartitionedCallЂ$block5_conv3/StatefulPartitionedCallЂfc1/StatefulPartitionedCallЂfc2/StatefulPartitionedCallЂ#predictions/StatefulPartitionedCallв
lambda/PartitionedCallPartitionedCallinputs#lambda_partitionedcall_lambda_const*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*1
_output_shapes
:џџџџџџџџџрр*,
_gradient_op_typePartitionedCall-19019*J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_19006*
Tout
2Н
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0+block1_conv1_statefulpartitionedcall_args_1+block1_conv1_statefulpartitionedcall_args_2*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*1
_output_shapes
:џџџџџџџџџрр@*
Tin
2*,
_gradient_op_typePartitionedCall-18570*P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_18564Ы
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0+block1_conv2_statefulpartitionedcall_args_1+block1_conv2_statefulpartitionedcall_args_2*1
_output_shapes
:џџџџџџџџџрр@*
Tin
2*,
_gradient_op_typePartitionedCall-18595*P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_18589*
Tout
2*/
config_proto

CPU

GPU2 *0J 8л
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџpp@*,
_gradient_op_typePartitionedCall-18614*O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_18608*
Tout
2С
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0+block2_conv1_statefulpartitionedcall_args_1+block2_conv1_statefulpartitionedcall_args_2*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*0
_output_shapes
:џџџџџџџџџpp*
Tin
2*,
_gradient_op_typePartitionedCall-18637*P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_18631Ъ
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0+block2_conv2_statefulpartitionedcall_args_1+block2_conv2_statefulpartitionedcall_args_2*/
config_proto

CPU

GPU2 *0J 8*0
_output_shapes
:џџџџџџџџџpp*
Tin
2*,
_gradient_op_typePartitionedCall-18662*P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_18656*
Tout
2м
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџ88*,
_gradient_op_typePartitionedCall-18681*O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_18675*
Tout
2С
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0+block3_conv1_statefulpartitionedcall_args_1+block3_conv1_statefulpartitionedcall_args_2*P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_18698*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*0
_output_shapes
:џџџџџџџџџ88*
Tin
2*,
_gradient_op_typePartitionedCall-18704Ъ
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0+block3_conv2_statefulpartitionedcall_args_1+block3_conv2_statefulpartitionedcall_args_2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџ88*,
_gradient_op_typePartitionedCall-18729*P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_18723*
Tout
2Ъ
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0+block3_conv3_statefulpartitionedcall_args_1+block3_conv3_statefulpartitionedcall_args_2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџ88*,
_gradient_op_typePartitionedCall-18754*P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_18748*
Tout
2м
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*0
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-18773*O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_18767*
Tout
2*/
config_proto

CPU

GPU2 *0J 8С
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0+block4_conv1_statefulpartitionedcall_args_1+block4_conv1_statefulpartitionedcall_args_2*P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_18790*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-18796Ъ
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0+block4_conv2_statefulpartitionedcall_args_1+block4_conv2_statefulpartitionedcall_args_2*0
_output_shapes
:џџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-18821*P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_18815*
Tout
2*/
config_proto

CPU

GPU2 *0J 8Ъ
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0+block4_conv3_statefulpartitionedcall_args_1+block4_conv3_statefulpartitionedcall_args_2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-18846*P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_18840*
Tout
2м
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*0
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-18865*O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_18859*
Tout
2*/
config_proto

CPU

GPU2 *0J 8С
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0+block5_conv1_statefulpartitionedcall_args_1+block5_conv1_statefulpartitionedcall_args_2*P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_18882*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*0
_output_shapes
:џџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-18888Ъ
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0+block5_conv2_statefulpartitionedcall_args_1+block5_conv2_statefulpartitionedcall_args_2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-18913*P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_18907*
Tout
2Ъ
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0+block5_conv3_statefulpartitionedcall_args_1+block5_conv3_statefulpartitionedcall_args_2*0
_output_shapes
:џџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-18938*P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_18932*
Tout
2*/
config_proto

CPU

GPU2 *0J 8м
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-18957*O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_18951*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџх
(global_average_pooling2d/PartitionedCallPartitionedCall$block5_pool/PartitionedCall:output:0*,
_gradient_op_typePartitionedCall-18975*\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_18969*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*(
_output_shapes
:џџџџџџџџџ*
Tin
2Ђ
fc1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0"fc1_statefulpartitionedcall_args_1"fc1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-19088*G
fBR@
>__inference_fc1_layer_call_and_return_conditional_losses_19082*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*(
_output_shapes
:џџџџџџџџџ
fc2/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0"fc2_statefulpartitionedcall_args_1"fc2_statefulpartitionedcall_args_2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*'
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-19116*G
fBR@
>__inference_fc2_layer_call_and_return_conditional_losses_19110*
Tout
2Д
#predictions/StatefulPartitionedCallStatefulPartitionedCall$fc2/StatefulPartitionedCall:output:0*predictions_statefulpartitionedcall_args_1*predictions_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-19144*O
fJRH
F__inference_predictions_layer_call_and_return_conditional_losses_19138*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*'
_output_shapes
:џџџџџџџџџ*
Tin
2б
IdentityIdentity,predictions/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall$^predictions/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*И
_input_shapesІ
Ѓ:џџџџџџџџџрр:::::::::::::::::::::::::::::::::2J
#predictions/StatefulPartitionedCall#predictions/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! 

w
A__inference_lambda_layer_call_and_return_conditional_losses_18994

inputs
biasadd_lambda_const
identityd
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        f
strided_slice/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_2Const*
valueB"   џџџџ*
dtype0*
_output_shapes
:
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*

begin_mask*
ellipsis_mask*
end_mask*1
_output_shapes
:џџџџџџџџџрр|
BiasAddBiasAddstrided_slice:output:0biasadd_lambda_const*1
_output_shapes
:џџџџџџџџџрр*
T0b
IdentityIdentityBiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџрр"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџрр::& "
 
_user_specified_nameinputs: 
Г
­
,__inference_block4_conv1_layer_call_fn_18801

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-18796*P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_18790*
Tout
2*/
config_proto

CPU

GPU2 *0J 8
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs

d
&__inference_lambda_layer_call_fn_19835

inputs 
partitionedcall_lambda_const
identityФ
PartitionedCallPartitionedCallinputspartitionedcall_lambda_const*,
_gradient_op_typePartitionedCall-19019*J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_19006*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*1
_output_shapes
:џџџџџџџџџрр*
Tin
2j
IdentityIdentityPartitionedCall:output:0*1
_output_shapes
:џџџџџџџџџрр*
T0"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџрр::& "
 
_user_specified_nameinputs: 

р
G__inference_block5_conv3_layer_call_and_return_conditional_losses_18932

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:Ќ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџЁ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0k
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџІ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
ўo
р
@__inference_model_layer_call_and_return_conditional_losses_19218
input_1'
#lambda_partitionedcall_lambda_const/
+block1_conv1_statefulpartitionedcall_args_1/
+block1_conv1_statefulpartitionedcall_args_2/
+block1_conv2_statefulpartitionedcall_args_1/
+block1_conv2_statefulpartitionedcall_args_2/
+block2_conv1_statefulpartitionedcall_args_1/
+block2_conv1_statefulpartitionedcall_args_2/
+block2_conv2_statefulpartitionedcall_args_1/
+block2_conv2_statefulpartitionedcall_args_2/
+block3_conv1_statefulpartitionedcall_args_1/
+block3_conv1_statefulpartitionedcall_args_2/
+block3_conv2_statefulpartitionedcall_args_1/
+block3_conv2_statefulpartitionedcall_args_2/
+block3_conv3_statefulpartitionedcall_args_1/
+block3_conv3_statefulpartitionedcall_args_2/
+block4_conv1_statefulpartitionedcall_args_1/
+block4_conv1_statefulpartitionedcall_args_2/
+block4_conv2_statefulpartitionedcall_args_1/
+block4_conv2_statefulpartitionedcall_args_2/
+block4_conv3_statefulpartitionedcall_args_1/
+block4_conv3_statefulpartitionedcall_args_2/
+block5_conv1_statefulpartitionedcall_args_1/
+block5_conv1_statefulpartitionedcall_args_2/
+block5_conv2_statefulpartitionedcall_args_1/
+block5_conv2_statefulpartitionedcall_args_2/
+block5_conv3_statefulpartitionedcall_args_1/
+block5_conv3_statefulpartitionedcall_args_2&
"fc1_statefulpartitionedcall_args_1&
"fc1_statefulpartitionedcall_args_2&
"fc2_statefulpartitionedcall_args_1&
"fc2_statefulpartitionedcall_args_2.
*predictions_statefulpartitionedcall_args_1.
*predictions_statefulpartitionedcall_args_2
identityЂ$block1_conv1/StatefulPartitionedCallЂ$block1_conv2/StatefulPartitionedCallЂ$block2_conv1/StatefulPartitionedCallЂ$block2_conv2/StatefulPartitionedCallЂ$block3_conv1/StatefulPartitionedCallЂ$block3_conv2/StatefulPartitionedCallЂ$block3_conv3/StatefulPartitionedCallЂ$block4_conv1/StatefulPartitionedCallЂ$block4_conv2/StatefulPartitionedCallЂ$block4_conv3/StatefulPartitionedCallЂ$block5_conv1/StatefulPartitionedCallЂ$block5_conv2/StatefulPartitionedCallЂ$block5_conv3/StatefulPartitionedCallЂfc1/StatefulPartitionedCallЂfc2/StatefulPartitionedCallЂ#predictions/StatefulPartitionedCallг
lambda/PartitionedCallPartitionedCallinput_1#lambda_partitionedcall_lambda_const*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*1
_output_shapes
:џџџџџџџџџрр*,
_gradient_op_typePartitionedCall-19019*J
fERC
A__inference_lambda_layer_call_and_return_conditional_losses_19006Н
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0+block1_conv1_statefulpartitionedcall_args_1+block1_conv1_statefulpartitionedcall_args_2*1
_output_shapes
:џџџџџџџџџрр@*
Tin
2*,
_gradient_op_typePartitionedCall-18570*P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_18564*
Tout
2*/
config_proto

CPU

GPU2 *0J 8Ы
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0+block1_conv2_statefulpartitionedcall_args_1+block1_conv2_statefulpartitionedcall_args_2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*1
_output_shapes
:џџџџџџџџџрр@*,
_gradient_op_typePartitionedCall-18595*P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_18589*
Tout
2л
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*/
_output_shapes
:џџџџџџџџџpp@*,
_gradient_op_typePartitionedCall-18614*O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_18608*
Tout
2С
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0+block2_conv1_statefulpartitionedcall_args_1+block2_conv1_statefulpartitionedcall_args_2*P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_18631*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџpp*,
_gradient_op_typePartitionedCall-18637Ъ
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0+block2_conv2_statefulpartitionedcall_args_1+block2_conv2_statefulpartitionedcall_args_2*0
_output_shapes
:џџџџџџџџџpp*
Tin
2*,
_gradient_op_typePartitionedCall-18662*P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_18656*
Tout
2*/
config_proto

CPU

GPU2 *0J 8м
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџ88*,
_gradient_op_typePartitionedCall-18681*O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_18675С
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0+block3_conv1_statefulpartitionedcall_args_1+block3_conv1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-18704*P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_18698*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџ88Ъ
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0+block3_conv2_statefulpartitionedcall_args_1+block3_conv2_statefulpartitionedcall_args_2*P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_18723*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*0
_output_shapes
:џџџџџџџџџ88*
Tin
2*,
_gradient_op_typePartitionedCall-18729Ъ
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0+block3_conv3_statefulpartitionedcall_args_1+block3_conv3_statefulpartitionedcall_args_2*
Tin
2*0
_output_shapes
:џџџџџџџџџ88*,
_gradient_op_typePartitionedCall-18754*P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_18748*
Tout
2*/
config_proto

CPU

GPU2 *0J 8м
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_18767*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-18773С
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0+block4_conv1_statefulpartitionedcall_args_1+block4_conv1_statefulpartitionedcall_args_2*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-18796*P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_18790Ъ
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0+block4_conv2_statefulpartitionedcall_args_1+block4_conv2_statefulpartitionedcall_args_2*P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_18815*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-18821Ъ
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0+block4_conv3_statefulpartitionedcall_args_1+block4_conv3_statefulpartitionedcall_args_2*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*0
_output_shapes
:џџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-18846*P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_18840м
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-18865*O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_18859С
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0+block5_conv1_statefulpartitionedcall_args_1+block5_conv1_statefulpartitionedcall_args_2*
Tin
2*0
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-18888*P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_18882*
Tout
2*/
config_proto

CPU

GPU2 *0J 8Ъ
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0+block5_conv2_statefulpartitionedcall_args_1+block5_conv2_statefulpartitionedcall_args_2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*0
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-18913*P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_18907*
Tout
2Ъ
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0+block5_conv3_statefulpartitionedcall_args_1+block5_conv3_statefulpartitionedcall_args_2*0
_output_shapes
:џџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-18938*P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_18932*
Tout
2*/
config_proto

CPU

GPU2 *0J 8м
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*/
config_proto

CPU

GPU2 *0J 8*0
_output_shapes
:џџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-18957*O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_18951*
Tout
2х
(global_average_pooling2d/PartitionedCallPartitionedCall$block5_pool/PartitionedCall:output:0*,
_gradient_op_typePartitionedCall-18975*\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_18969*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*(
_output_shapes
:џџџџџџџџџЂ
fc1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0"fc1_statefulpartitionedcall_args_1"fc1_statefulpartitionedcall_args_2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*(
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-19088*G
fBR@
>__inference_fc1_layer_call_and_return_conditional_losses_19082*
Tout
2
fc2/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0"fc2_statefulpartitionedcall_args_1"fc2_statefulpartitionedcall_args_2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*'
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-19116*G
fBR@
>__inference_fc2_layer_call_and_return_conditional_losses_19110*
Tout
2Д
#predictions/StatefulPartitionedCallStatefulPartitionedCall$fc2/StatefulPartitionedCall:output:0*predictions_statefulpartitionedcall_args_1*predictions_statefulpartitionedcall_args_2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*'
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-19144*O
fJRH
F__inference_predictions_layer_call_and_return_conditional_losses_19138*
Tout
2б
IdentityIdentity,predictions/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall$^predictions/StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*И
_input_shapesІ
Ѓ:џџџџџџџџџрр:::::::::::::::::::::::::::::::::2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2J
#predictions/StatefulPartitionedCall#predictions/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! 
Г
­
,__inference_block4_conv3_layer_call_fn_18851

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_18840*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*,
_gradient_op_typePartitionedCall-18846
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 

b
F__inference_block2_pool_layer_call_and_return_conditional_losses_18675

inputs
identityЂ
MaxPoolMaxPoolinputs*
ksize
*
paddingVALID*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:& "
 
_user_specified_nameinputs

b
F__inference_block5_pool_layer_call_and_return_conditional_losses_18951

inputs
identityЂ
MaxPoolMaxPoolinputs*
ksize
*
paddingVALID*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
strides
{
IdentityIdentityMaxPool:output:0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:& "
 
_user_specified_nameinputs

T
8__inference_global_average_pooling2d_layer_call_fn_18978

inputs
identityЖ
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-18975*\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_18969*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
Tin
2i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:& "
 
_user_specified_nameinputs
Ь	
з
>__inference_fc2_layer_call_and_return_conditional_losses_19864

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЃ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ*
T0 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*'
_output_shapes
:џџџџџџџџџ*
T0
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 

р
G__inference_block3_conv3_layer_call_and_return_conditional_losses_18748

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*(
_output_shapes
:*
dtype0Ќ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0*
strides
*
paddingSAMEЁ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџk
ReluReluBiasAdd:output:0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0І
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
в	
з
>__inference_fc1_layer_call_and_return_conditional_losses_19846

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЄ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЁ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 

р
G__inference_block1_conv2_layer_call_and_return_conditional_losses_18589

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЊ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@Ћ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingSAME*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
T0*
strides
 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
T0j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@Ѕ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: : :& "
 
_user_specified_nameinputs

р
G__inference_block5_conv2_layer_call_and_return_conditional_losses_18907

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:Ќ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0Ё
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџk
ReluReluBiasAdd:output:0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0І
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
Г
Ё

%__inference_model_layer_call_fn_19419
input_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33(
$statefulpartitionedcall_lambda_const
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1$statefulpartitionedcall_lambda_conststatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33*/
config_proto

CPU

GPU2 *0J 8*-
Tin&
$2"*'
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-19383*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_19382*
Tout
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*И
_input_shapesІ
Ѓ:џџџџџџџџџрр:::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : : : : : : : : : : : : : : :  :! :' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : 
А

@__inference_model_layer_call_and_return_conditional_losses_19598

inputs
lambda_biasadd_lambda_const/
+block1_conv1_conv2d_readvariableop_resource0
,block1_conv1_biasadd_readvariableop_resource/
+block1_conv2_conv2d_readvariableop_resource0
,block1_conv2_biasadd_readvariableop_resource/
+block2_conv1_conv2d_readvariableop_resource0
,block2_conv1_biasadd_readvariableop_resource/
+block2_conv2_conv2d_readvariableop_resource0
,block2_conv2_biasadd_readvariableop_resource/
+block3_conv1_conv2d_readvariableop_resource0
,block3_conv1_biasadd_readvariableop_resource/
+block3_conv2_conv2d_readvariableop_resource0
,block3_conv2_biasadd_readvariableop_resource/
+block3_conv3_conv2d_readvariableop_resource0
,block3_conv3_biasadd_readvariableop_resource/
+block4_conv1_conv2d_readvariableop_resource0
,block4_conv1_biasadd_readvariableop_resource/
+block4_conv2_conv2d_readvariableop_resource0
,block4_conv2_biasadd_readvariableop_resource/
+block4_conv3_conv2d_readvariableop_resource0
,block4_conv3_biasadd_readvariableop_resource/
+block5_conv1_conv2d_readvariableop_resource0
,block5_conv1_biasadd_readvariableop_resource/
+block5_conv2_conv2d_readvariableop_resource0
,block5_conv2_biasadd_readvariableop_resource/
+block5_conv3_conv2d_readvariableop_resource0
,block5_conv3_biasadd_readvariableop_resource&
"fc1_matmul_readvariableop_resource'
#fc1_biasadd_readvariableop_resource&
"fc2_matmul_readvariableop_resource'
#fc2_biasadd_readvariableop_resource.
*predictions_matmul_readvariableop_resource/
+predictions_biasadd_readvariableop_resource
identityЂ#block1_conv1/BiasAdd/ReadVariableOpЂ"block1_conv1/Conv2D/ReadVariableOpЂ#block1_conv2/BiasAdd/ReadVariableOpЂ"block1_conv2/Conv2D/ReadVariableOpЂ#block2_conv1/BiasAdd/ReadVariableOpЂ"block2_conv1/Conv2D/ReadVariableOpЂ#block2_conv2/BiasAdd/ReadVariableOpЂ"block2_conv2/Conv2D/ReadVariableOpЂ#block3_conv1/BiasAdd/ReadVariableOpЂ"block3_conv1/Conv2D/ReadVariableOpЂ#block3_conv2/BiasAdd/ReadVariableOpЂ"block3_conv2/Conv2D/ReadVariableOpЂ#block3_conv3/BiasAdd/ReadVariableOpЂ"block3_conv3/Conv2D/ReadVariableOpЂ#block4_conv1/BiasAdd/ReadVariableOpЂ"block4_conv1/Conv2D/ReadVariableOpЂ#block4_conv2/BiasAdd/ReadVariableOpЂ"block4_conv2/Conv2D/ReadVariableOpЂ#block4_conv3/BiasAdd/ReadVariableOpЂ"block4_conv3/Conv2D/ReadVariableOpЂ#block5_conv1/BiasAdd/ReadVariableOpЂ"block5_conv1/Conv2D/ReadVariableOpЂ#block5_conv2/BiasAdd/ReadVariableOpЂ"block5_conv2/Conv2D/ReadVariableOpЂ#block5_conv3/BiasAdd/ReadVariableOpЂ"block5_conv3/Conv2D/ReadVariableOpЂfc1/BiasAdd/ReadVariableOpЂfc1/MatMul/ReadVariableOpЂfc2/BiasAdd/ReadVariableOpЂfc2/MatMul/ReadVariableOpЂ"predictions/BiasAdd/ReadVariableOpЂ!predictions/MatMul/ReadVariableOpk
lambda/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:m
lambda/strided_slice/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0m
lambda/strided_slice/stack_2Const*
valueB"   џџџџ*
dtype0*
_output_shapes
:
lambda/strided_sliceStridedSliceinputs#lambda/strided_slice/stack:output:0%lambda/strided_slice/stack_1:output:0%lambda/strided_slice/stack_2:output:0*
Index0*
T0*
ellipsis_mask*

begin_mask*
end_mask*1
_output_shapes
:џџџџџџџџџрр
lambda/BiasAddBiasAddlambda/strided_slice:output:0lambda_biasadd_lambda_const*
T0*1
_output_shapes
:џџџџџџџџџррФ
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@Ц
block1_conv1/Conv2DConv2Dlambda/BiasAdd:output:0*block1_conv1/Conv2D/ReadVariableOp:value:0*1
_output_shapes
:џџџџџџџџџрр@*
T0*
strides
*
paddingSAMEК
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0І
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџрр@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџрр@Ф
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@Ю
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*1
_output_shapes
:џџџџџџџџџрр@*
T0*
strides
*
paddingSAMEК
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@І
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџрр@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџрр@Ќ
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџpp@*
strides
*
ksize
*
paddingVALIDХ
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*'
_output_shapes
:@*
dtype0Ъ
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџpp*
T0*
strides
*
paddingSAMEЛ
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:Ѕ
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџpp*
T0s
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџppЦ
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:Э
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:џџџџџџџџџppЛ
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:Ѕ
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџpp*
T0s
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџpp­
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:џџџџџџџџџ88Ц
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:Ъ
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:џџџџџџџџџ88Л
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:Ѕ
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ88s
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*0
_output_shapes
:џџџџџџџџџ88*
T0Ц
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:Э
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџ88*
T0*
strides
*
paddingSAMEЛ
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:Ѕ
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ88s
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ88Ц
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:Э
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:џџџџџџџџџ88*
T0Л
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:*
dtype0Ѕ
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџ88*
T0s
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*0
_output_shapes
:џџџџџџџџџ88*
T0­
block3_pool/MaxPoolMaxPoolblock3_conv3/Relu:activations:0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:џџџџџџџџџЦ
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:Ъ
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:џџџџџџџџџ*
T0*
strides
Л
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:Ѕ
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџs
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*0
_output_shapes
:џџџџџџџџџ*
T0Ц
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:Э
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:џџџџџџџџџЛ
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:Ѕ
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџs
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЦ
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:Э
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџ*
T0*
strides
*
paddingSAMEЛ
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:Ѕ
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџs
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*0
_output_shapes
:џџџџџџџџџ*
T0­
block4_pool/MaxPoolMaxPoolblock4_conv3/Relu:activations:0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:џџџџџџџџџЦ
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:Ъ
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:џџџџџџџџџ*
T0Л
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:Ѕ
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџ*
T0s
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*0
_output_shapes
:џџџџџџџџџ*
T0Ц
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:Э
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџ*
T0*
strides
*
paddingSAMEЛ
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:Ѕ
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџs
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЦ
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:Э
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:џџџџџџџџџ*
T0Л
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:Ѕ
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџs
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*0
_output_shapes
:џџџџџџџџџ*
T0­
block5_pool/MaxPoolMaxPoolblock5_conv3/Relu:activations:0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:џџџџџџџџџ
/global_average_pooling2d/Mean/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:А
global_average_pooling2d/MeanMeanblock5_pool/MaxPool:output:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
fc1/MatMul/ReadVariableOpReadVariableOp"fc1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:


fc1/MatMulMatMul&global_average_pooling2d/Mean:output:0!fc1/MatMul/ReadVariableOp:value:0*(
_output_shapes
:џџџџџџџџџ*
T0Љ
fc1/BiasAdd/ReadVariableOpReadVariableOp#fc1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:
fc1/BiasAddBiasAddfc1/MatMul:product:0"fc1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџY
fc1/ReluRelufc1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџЋ
fc2/MatMul/ReadVariableOpReadVariableOp"fc2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	

fc2/MatMulMatMulfc1/Relu:activations:0!fc2/MatMul/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ*
T0Ј
fc2/BiasAdd/ReadVariableOpReadVariableOp#fc2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:
fc2/BiasAddBiasAddfc2/MatMul:product:0"fc2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџX
fc2/ReluRelufc2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџК
!predictions/MatMul/ReadVariableOpReadVariableOp*predictions_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:
predictions/MatMulMatMulfc2/Relu:activations:0)predictions/MatMul/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ*
T0И
"predictions/BiasAdd/ReadVariableOpReadVariableOp+predictions_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:
predictions/BiasAddBiasAddpredictions/MatMul:product:0*predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџn
predictions/SoftmaxSoftmaxpredictions/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџя	
IdentityIdentitypredictions/Softmax:softmax:0$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp^fc1/BiasAdd/ReadVariableOp^fc1/MatMul/ReadVariableOp^fc2/BiasAdd/ReadVariableOp^fc2/MatMul/ReadVariableOp#^predictions/BiasAdd/ReadVariableOp"^predictions/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*И
_input_shapesІ
Ѓ:џџџџџџџџџрр:::::::::::::::::::::::::::::::::2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp28
fc2/BiasAdd/ReadVariableOpfc2/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2F
!predictions/MatMul/ReadVariableOp!predictions/MatMul/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp28
fc1/BiasAdd/ReadVariableOpfc1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp26
fc1/MatMul/ReadVariableOpfc1/MatMul/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp26
fc2/MatMul/ReadVariableOpfc2/MatMul/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"predictions/BiasAdd/ReadVariableOp"predictions/BiasAdd/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp: : : : : : : : : : :  :! :& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : 
Г
­
,__inference_block3_conv2_layer_call_fn_18734

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-18729*P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_18723*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
Tin
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
В
­
,__inference_block2_conv1_layer_call_fn_18642

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*,
_gradient_op_typePartitionedCall-18637*P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_18631*
Tout
2*/
config_proto

CPU

GPU2 *0J 8
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 

р
G__inference_block5_conv1_layer_call_and_return_conditional_losses_18882

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:Ќ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingSAME*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0*
strides
Ё
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0k
ReluReluBiasAdd:output:0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0І
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
П
В
!__inference__traced_restore_20234
file_prefix(
$assignvariableop_block1_conv1_kernel(
$assignvariableop_1_block1_conv1_bias*
&assignvariableop_2_block1_conv2_kernel(
$assignvariableop_3_block1_conv2_bias*
&assignvariableop_4_block2_conv1_kernel(
$assignvariableop_5_block2_conv1_bias*
&assignvariableop_6_block2_conv2_kernel(
$assignvariableop_7_block2_conv2_bias*
&assignvariableop_8_block3_conv1_kernel(
$assignvariableop_9_block3_conv1_bias+
'assignvariableop_10_block3_conv2_kernel)
%assignvariableop_11_block3_conv2_bias+
'assignvariableop_12_block3_conv3_kernel)
%assignvariableop_13_block3_conv3_bias+
'assignvariableop_14_block4_conv1_kernel)
%assignvariableop_15_block4_conv1_bias+
'assignvariableop_16_block4_conv2_kernel)
%assignvariableop_17_block4_conv2_bias+
'assignvariableop_18_block4_conv3_kernel)
%assignvariableop_19_block4_conv3_bias+
'assignvariableop_20_block5_conv1_kernel)
%assignvariableop_21_block5_conv1_bias+
'assignvariableop_22_block5_conv2_kernel)
%assignvariableop_23_block5_conv2_bias+
'assignvariableop_24_block5_conv3_kernel)
%assignvariableop_25_block5_conv3_bias"
assignvariableop_26_fc1_kernel 
assignvariableop_27_fc1_bias"
assignvariableop_28_fc2_kernel 
assignvariableop_29_fc2_bias*
&assignvariableop_30_predictions_kernel(
$assignvariableop_31_predictions_bias*
&assignvariableop_32_training_adam_iter,
(assignvariableop_33_training_adam_beta_1,
(assignvariableop_34_training_adam_beta_2+
'assignvariableop_35_training_adam_decay3
/assignvariableop_36_training_adam_learning_rate
assignvariableop_37_total
assignvariableop_38_count2
.assignvariableop_39_training_adam_fc1_kernel_m0
,assignvariableop_40_training_adam_fc1_bias_m2
.assignvariableop_41_training_adam_fc2_kernel_m0
,assignvariableop_42_training_adam_fc2_bias_m:
6assignvariableop_43_training_adam_predictions_kernel_m8
4assignvariableop_44_training_adam_predictions_bias_m2
.assignvariableop_45_training_adam_fc1_kernel_v0
,assignvariableop_46_training_adam_fc1_bias_v2
.assignvariableop_47_training_adam_fc2_kernel_v0
,assignvariableop_48_training_adam_fc2_bias_v:
6assignvariableop_49_training_adam_predictions_kernel_v8
4assignvariableop_50_training_adam_predictions_bias_v
identity_52ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ђ	RestoreV2ЂRestoreV2_1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:3*Т
valueИBЕ3B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0ж
RestoreV2/shape_and_slicesConst"/device:CPU:0*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:3 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*т
_output_shapesЯ
Ь:::::::::::::::::::::::::::::::::::::::::::::::::::*A
dtypes7
523	L
IdentityIdentityRestoreV2:tensors:0*
_output_shapes
:*
T0
AssignVariableOpAssignVariableOp$assignvariableop_block1_conv1_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
_output_shapes
:*
T0
AssignVariableOp_1AssignVariableOp$assignvariableop_1_block1_conv1_biasIdentity_1:output:0*
_output_shapes
 *
dtype0N

Identity_2IdentityRestoreV2:tensors:2*
_output_shapes
:*
T0
AssignVariableOp_2AssignVariableOp&assignvariableop_2_block1_conv2_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp$assignvariableop_3_block1_conv2_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp&assignvariableop_4_block2_conv1_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp$assignvariableop_5_block2_conv1_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp&assignvariableop_6_block2_conv2_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp$assignvariableop_7_block2_conv2_biasIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
_output_shapes
:*
T0
AssignVariableOp_8AssignVariableOp&assignvariableop_8_block3_conv1_kernelIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
_output_shapes
:*
T0
AssignVariableOp_9AssignVariableOp$assignvariableop_9_block3_conv1_biasIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp'assignvariableop_10_block3_conv2_kernelIdentity_10:output:0*
_output_shapes
 *
dtype0P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp%assignvariableop_11_block3_conv2_biasIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp'assignvariableop_12_block3_conv3_kernelIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp%assignvariableop_13_block3_conv3_biasIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
_output_shapes
:*
T0
AssignVariableOp_14AssignVariableOp'assignvariableop_14_block4_conv1_kernelIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
_output_shapes
:*
T0
AssignVariableOp_15AssignVariableOp%assignvariableop_15_block4_conv1_biasIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp'assignvariableop_16_block4_conv2_kernelIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp%assignvariableop_17_block4_conv2_biasIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
_output_shapes
:*
T0
AssignVariableOp_18AssignVariableOp'assignvariableop_18_block4_conv3_kernelIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
_output_shapes
:*
T0
AssignVariableOp_19AssignVariableOp%assignvariableop_19_block4_conv3_biasIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp'assignvariableop_20_block5_conv1_kernelIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp%assignvariableop_21_block5_conv1_biasIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
_output_shapes
:*
T0
AssignVariableOp_22AssignVariableOp'assignvariableop_22_block5_conv2_kernelIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp%assignvariableop_23_block5_conv2_biasIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
_output_shapes
:*
T0
AssignVariableOp_24AssignVariableOp'assignvariableop_24_block5_conv3_kernelIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
_output_shapes
:*
T0
AssignVariableOp_25AssignVariableOp%assignvariableop_25_block5_conv3_biasIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
_output_shapes
:*
T0
AssignVariableOp_26AssignVariableOpassignvariableop_26_fc1_kernelIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:~
AssignVariableOp_27AssignVariableOpassignvariableop_27_fc1_biasIdentity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOpassignvariableop_28_fc2_kernelIdentity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
_output_shapes
:*
T0~
AssignVariableOp_29AssignVariableOpassignvariableop_29_fc2_biasIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp&assignvariableop_30_predictions_kernelIdentity_30:output:0*
dtype0*
_output_shapes
 P
Identity_31IdentityRestoreV2:tensors:31*
_output_shapes
:*
T0
AssignVariableOp_31AssignVariableOp$assignvariableop_31_predictions_biasIdentity_31:output:0*
_output_shapes
 *
dtype0P
Identity_32IdentityRestoreV2:tensors:32*
_output_shapes
:*
T0	
AssignVariableOp_32AssignVariableOp&assignvariableop_32_training_adam_iterIdentity_32:output:0*
dtype0	*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp(assignvariableop_33_training_adam_beta_1Identity_33:output:0*
dtype0*
_output_shapes
 P
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp(assignvariableop_34_training_adam_beta_2Identity_34:output:0*
dtype0*
_output_shapes
 P
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp'assignvariableop_35_training_adam_decayIdentity_35:output:0*
dtype0*
_output_shapes
 P
Identity_36IdentityRestoreV2:tensors:36*
_output_shapes
:*
T0
AssignVariableOp_36AssignVariableOp/assignvariableop_36_training_adam_learning_rateIdentity_36:output:0*
dtype0*
_output_shapes
 P
Identity_37IdentityRestoreV2:tensors:37*
_output_shapes
:*
T0{
AssignVariableOp_37AssignVariableOpassignvariableop_37_totalIdentity_37:output:0*
dtype0*
_output_shapes
 P
Identity_38IdentityRestoreV2:tensors:38*
_output_shapes
:*
T0{
AssignVariableOp_38AssignVariableOpassignvariableop_38_countIdentity_38:output:0*
dtype0*
_output_shapes
 P
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp.assignvariableop_39_training_adam_fc1_kernel_mIdentity_39:output:0*
dtype0*
_output_shapes
 P
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp,assignvariableop_40_training_adam_fc1_bias_mIdentity_40:output:0*
dtype0*
_output_shapes
 P
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp.assignvariableop_41_training_adam_fc2_kernel_mIdentity_41:output:0*
dtype0*
_output_shapes
 P
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp,assignvariableop_42_training_adam_fc2_bias_mIdentity_42:output:0*
dtype0*
_output_shapes
 P
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp6assignvariableop_43_training_adam_predictions_kernel_mIdentity_43:output:0*
dtype0*
_output_shapes
 P
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp4assignvariableop_44_training_adam_predictions_bias_mIdentity_44:output:0*
dtype0*
_output_shapes
 P
Identity_45IdentityRestoreV2:tensors:45*
_output_shapes
:*
T0
AssignVariableOp_45AssignVariableOp.assignvariableop_45_training_adam_fc1_kernel_vIdentity_45:output:0*
dtype0*
_output_shapes
 P
Identity_46IdentityRestoreV2:tensors:46*
_output_shapes
:*
T0
AssignVariableOp_46AssignVariableOp,assignvariableop_46_training_adam_fc1_bias_vIdentity_46:output:0*
dtype0*
_output_shapes
 P
Identity_47IdentityRestoreV2:tensors:47*
_output_shapes
:*
T0
AssignVariableOp_47AssignVariableOp.assignvariableop_47_training_adam_fc2_kernel_vIdentity_47:output:0*
dtype0*
_output_shapes
 P
Identity_48IdentityRestoreV2:tensors:48*
_output_shapes
:*
T0
AssignVariableOp_48AssignVariableOp,assignvariableop_48_training_adam_fc2_bias_vIdentity_48:output:0*
dtype0*
_output_shapes
 P
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp6assignvariableop_49_training_adam_predictions_kernel_vIdentity_49:output:0*
dtype0*
_output_shapes
 P
Identity_50IdentityRestoreV2:tensors:50*
_output_shapes
:*
T0
AssignVariableOp_50AssignVariableOp4assignvariableop_50_training_adam_predictions_bias_vIdentity_50:output:0*
dtype0*
_output_shapes
 
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:Е
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 Б	
Identity_51Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: О	
Identity_52IdentityIdentity_51:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"#
identity_52Identity_52:output:0*у
_input_shapesб
Ю: :::::::::::::::::::::::::::::::::::::::::::::::::::2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_27AssignVariableOp_272$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_39AssignVariableOp_392*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 

w
A__inference_lambda_layer_call_and_return_conditional_losses_19823

inputs
biasadd_lambda_const
identityd
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"        f
strided_slice/stack_2Const*
valueB"   џџџџ*
dtype0*
_output_shapes
:
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*1
_output_shapes
:џџџџџџџџџрр*
T0*
Index0*

begin_mask*
ellipsis_mask*
end_mask|
BiasAddBiasAddstrided_slice:output:0biasadd_lambda_const*
T0*1
_output_shapes
:џџџџџџџџџррb
IdentityIdentityBiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџрр"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџрр::& "
 
_user_specified_nameinputs: 
А
 

%__inference_model_layer_call_fn_19765

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33(
$statefulpartitionedcall_lambda_const
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs$statefulpartitionedcall_lambda_conststatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33*/
config_proto

CPU

GPU2 *0J 8*-
Tin&
$2"*'
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-19282*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_19281*
Tout
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*И
_input_shapesІ
Ѓ:џџџџџџџџџрр:::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : : : : : : : : :  :! :& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : 
Г
­
,__inference_block4_conv2_layer_call_fn_18826

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_18815*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-18821
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
з	
п
F__inference_predictions_layer_call_and_return_conditional_losses_19882

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
СС
ћ
 __inference__wrapped_model_18550
input_1%
!model_lambda_biasadd_lambda_const5
1model_block1_conv1_conv2d_readvariableop_resource6
2model_block1_conv1_biasadd_readvariableop_resource5
1model_block1_conv2_conv2d_readvariableop_resource6
2model_block1_conv2_biasadd_readvariableop_resource5
1model_block2_conv1_conv2d_readvariableop_resource6
2model_block2_conv1_biasadd_readvariableop_resource5
1model_block2_conv2_conv2d_readvariableop_resource6
2model_block2_conv2_biasadd_readvariableop_resource5
1model_block3_conv1_conv2d_readvariableop_resource6
2model_block3_conv1_biasadd_readvariableop_resource5
1model_block3_conv2_conv2d_readvariableop_resource6
2model_block3_conv2_biasadd_readvariableop_resource5
1model_block3_conv3_conv2d_readvariableop_resource6
2model_block3_conv3_biasadd_readvariableop_resource5
1model_block4_conv1_conv2d_readvariableop_resource6
2model_block4_conv1_biasadd_readvariableop_resource5
1model_block4_conv2_conv2d_readvariableop_resource6
2model_block4_conv2_biasadd_readvariableop_resource5
1model_block4_conv3_conv2d_readvariableop_resource6
2model_block4_conv3_biasadd_readvariableop_resource5
1model_block5_conv1_conv2d_readvariableop_resource6
2model_block5_conv1_biasadd_readvariableop_resource5
1model_block5_conv2_conv2d_readvariableop_resource6
2model_block5_conv2_biasadd_readvariableop_resource5
1model_block5_conv3_conv2d_readvariableop_resource6
2model_block5_conv3_biasadd_readvariableop_resource,
(model_fc1_matmul_readvariableop_resource-
)model_fc1_biasadd_readvariableop_resource,
(model_fc2_matmul_readvariableop_resource-
)model_fc2_biasadd_readvariableop_resource4
0model_predictions_matmul_readvariableop_resource5
1model_predictions_biasadd_readvariableop_resource
identityЂ)model/block1_conv1/BiasAdd/ReadVariableOpЂ(model/block1_conv1/Conv2D/ReadVariableOpЂ)model/block1_conv2/BiasAdd/ReadVariableOpЂ(model/block1_conv2/Conv2D/ReadVariableOpЂ)model/block2_conv1/BiasAdd/ReadVariableOpЂ(model/block2_conv1/Conv2D/ReadVariableOpЂ)model/block2_conv2/BiasAdd/ReadVariableOpЂ(model/block2_conv2/Conv2D/ReadVariableOpЂ)model/block3_conv1/BiasAdd/ReadVariableOpЂ(model/block3_conv1/Conv2D/ReadVariableOpЂ)model/block3_conv2/BiasAdd/ReadVariableOpЂ(model/block3_conv2/Conv2D/ReadVariableOpЂ)model/block3_conv3/BiasAdd/ReadVariableOpЂ(model/block3_conv3/Conv2D/ReadVariableOpЂ)model/block4_conv1/BiasAdd/ReadVariableOpЂ(model/block4_conv1/Conv2D/ReadVariableOpЂ)model/block4_conv2/BiasAdd/ReadVariableOpЂ(model/block4_conv2/Conv2D/ReadVariableOpЂ)model/block4_conv3/BiasAdd/ReadVariableOpЂ(model/block4_conv3/Conv2D/ReadVariableOpЂ)model/block5_conv1/BiasAdd/ReadVariableOpЂ(model/block5_conv1/Conv2D/ReadVariableOpЂ)model/block5_conv2/BiasAdd/ReadVariableOpЂ(model/block5_conv2/Conv2D/ReadVariableOpЂ)model/block5_conv3/BiasAdd/ReadVariableOpЂ(model/block5_conv3/Conv2D/ReadVariableOpЂ model/fc1/BiasAdd/ReadVariableOpЂmodel/fc1/MatMul/ReadVariableOpЂ model/fc2/BiasAdd/ReadVariableOpЂmodel/fc2/MatMul/ReadVariableOpЂ(model/predictions/BiasAdd/ReadVariableOpЂ'model/predictions/MatMul/ReadVariableOpq
 model/lambda/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:s
"model/lambda/strided_slice/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:s
"model/lambda/strided_slice/stack_2Const*
_output_shapes
:*
valueB"   џџџџ*
dtype0И
model/lambda/strided_sliceStridedSliceinput_1)model/lambda/strided_slice/stack:output:0+model/lambda/strided_slice/stack_1:output:0+model/lambda/strided_slice/stack_2:output:0*
T0*
Index0*
ellipsis_mask*

begin_mask*
end_mask*1
_output_shapes
:џџџџџџџџџррЃ
model/lambda/BiasAddBiasAdd#model/lambda/strided_slice:output:0!model_lambda_biasadd_lambda_const*
T0*1
_output_shapes
:џџџџџџџџџрра
(model/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block1_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@и
model/block1_conv1/Conv2DConv2Dmodel/lambda/BiasAdd:output:00model/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*1
_output_shapes
:џџџџџџџџџрр@Ц
)model/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block1_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@И
model/block1_conv1/BiasAddBiasAdd"model/block1_conv1/Conv2D:output:01model/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџрр@
model/block1_conv1/ReluRelu#model/block1_conv1/BiasAdd:output:0*1
_output_shapes
:џџџџџџџџџрр@*
T0а
(model/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block1_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:@@*
dtype0р
model/block1_conv2/Conv2DConv2D%model/block1_conv1/Relu:activations:00model/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*1
_output_shapes
:џџџџџџџџџрр@Ц
)model/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block1_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0И
model/block1_conv2/BiasAddBiasAdd"model/block1_conv2/Conv2D:output:01model/block1_conv2/BiasAdd/ReadVariableOp:value:0*1
_output_shapes
:џџџџџџџџџрр@*
T0
model/block1_conv2/ReluRelu#model/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџрр@И
model/block1_pool/MaxPoolMaxPool%model/block1_conv2/Relu:activations:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:џџџџџџџџџpp@б
(model/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block2_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*'
_output_shapes
:@м
model/block2_conv1/Conv2DConv2D"model/block1_pool/MaxPool:output:00model/block2_conv1/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:џџџџџџџџџpp*
T0*
strides
Ч
)model/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block2_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:З
model/block2_conv1/BiasAddBiasAdd"model/block2_conv1/Conv2D:output:01model/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџpp
model/block2_conv1/ReluRelu#model/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџppв
(model/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block2_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:п
model/block2_conv2/Conv2DConv2D%model/block2_conv1/Relu:activations:00model/block2_conv2/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџpp*
T0*
strides
*
paddingSAMEЧ
)model/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block2_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:З
model/block2_conv2/BiasAddBiasAdd"model/block2_conv2/Conv2D:output:01model/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџpp
model/block2_conv2/ReluRelu#model/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџppЙ
model/block2_pool/MaxPoolMaxPool%model/block2_conv2/Relu:activations:0*
ksize
*
paddingVALID*0
_output_shapes
:џџџџџџџџџ88*
strides
в
(model/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block3_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:м
model/block3_conv1/Conv2DConv2D"model/block2_pool/MaxPool:output:00model/block3_conv1/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:џџџџџџџџџ88*
T0*
strides
Ч
)model/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block3_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:З
model/block3_conv1/BiasAddBiasAdd"model/block3_conv1/Conv2D:output:01model/block3_conv1/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџ88*
T0
model/block3_conv1/ReluRelu#model/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ88в
(model/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block3_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*(
_output_shapes
:*
dtype0п
model/block3_conv2/Conv2DConv2D%model/block3_conv1/Relu:activations:00model/block3_conv2/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:џџџџџџџџџ88*
T0*
strides
Ч
)model/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block3_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:З
model/block3_conv2/BiasAddBiasAdd"model/block3_conv2/Conv2D:output:01model/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ88
model/block3_conv2/ReluRelu#model/block3_conv2/BiasAdd:output:0*0
_output_shapes
:џџџџџџџџџ88*
T0в
(model/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1model_block3_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:п
model/block3_conv3/Conv2DConv2D%model/block3_conv2/Relu:activations:00model/block3_conv3/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:џџџџџџџџџ88*
T0*
strides
Ч
)model/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2model_block3_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:З
model/block3_conv3/BiasAddBiasAdd"model/block3_conv3/Conv2D:output:01model/block3_conv3/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџ88*
T0
model/block3_conv3/ReluRelu#model/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ88Й
model/block3_pool/MaxPoolMaxPool%model/block3_conv3/Relu:activations:0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:џџџџџџџџџв
(model/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block4_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:м
model/block4_conv1/Conv2DConv2D"model/block3_pool/MaxPool:output:00model/block4_conv1/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:џџџџџџџџџ*
T0Ч
)model/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block4_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:З
model/block4_conv1/BiasAddBiasAdd"model/block4_conv1/Conv2D:output:01model/block4_conv1/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџ*
T0
model/block4_conv1/ReluRelu#model/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџв
(model/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block4_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:п
model/block4_conv2/Conv2DConv2D%model/block4_conv1/Relu:activations:00model/block4_conv2/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:џџџџџџџџџ*
T0*
strides
Ч
)model/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block4_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:З
model/block4_conv2/BiasAddBiasAdd"model/block4_conv2/Conv2D:output:01model/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
model/block4_conv2/ReluRelu#model/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџв
(model/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1model_block4_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:п
model/block4_conv3/Conv2DConv2D%model/block4_conv2/Relu:activations:00model/block4_conv3/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџ*
T0*
strides
*
paddingSAMEЧ
)model/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2model_block4_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:З
model/block4_conv3/BiasAddBiasAdd"model/block4_conv3/Conv2D:output:01model/block4_conv3/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџ*
T0
model/block4_conv3/ReluRelu#model/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЙ
model/block4_pool/MaxPoolMaxPool%model/block4_conv3/Relu:activations:0*
ksize
*
paddingVALID*0
_output_shapes
:џџџџџџџџџ*
strides
в
(model/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1model_block5_conv1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:м
model/block5_conv1/Conv2DConv2D"model/block4_pool/MaxPool:output:00model/block5_conv1/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџ*
T0*
strides
*
paddingSAMEЧ
)model/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2model_block5_conv1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:*
dtype0З
model/block5_conv1/BiasAddBiasAdd"model/block5_conv1/Conv2D:output:01model/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
model/block5_conv1/ReluRelu#model/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџв
(model/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1model_block5_conv2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:п
model/block5_conv2/Conv2DConv2D%model/block5_conv1/Relu:activations:00model/block5_conv2/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџ*
T0*
strides
*
paddingSAMEЧ
)model/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2model_block5_conv2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:З
model/block5_conv2/BiasAddBiasAdd"model/block5_conv2/Conv2D:output:01model/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
model/block5_conv2/ReluRelu#model/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџв
(model/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1model_block5_conv3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:п
model/block5_conv3/Conv2DConv2D%model/block5_conv2/Relu:activations:00model/block5_conv3/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:џџџџџџџџџ*
T0*
strides
*
paddingSAMEЧ
)model/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2model_block5_conv3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:З
model/block5_conv3/BiasAddBiasAdd"model/block5_conv3/Conv2D:output:01model/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
model/block5_conv3/ReluRelu#model/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЙ
model/block5_pool/MaxPoolMaxPool%model/block5_conv3/Relu:activations:0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:џџџџџџџџџ
5model/global_average_pooling2d/Mean/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:Т
#model/global_average_pooling2d/MeanMean"model/block5_pool/MaxPool:output:0>model/global_average_pooling2d/Mean/reduction_indices:output:0*(
_output_shapes
:џџџџџџџџџ*
T0И
model/fc1/MatMul/ReadVariableOpReadVariableOp(model_fc1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
Є
model/fc1/MatMulMatMul,model/global_average_pooling2d/Mean:output:0'model/fc1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЕ
 model/fc1/BiasAdd/ReadVariableOpReadVariableOp)model_fc1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:
model/fc1/BiasAddBiasAddmodel/fc1/MatMul:product:0(model/fc1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџe
model/fc1/ReluRelumodel/fc1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџЗ
model/fc2/MatMul/ReadVariableOpReadVariableOp(model_fc2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	
model/fc2/MatMulMatMulmodel/fc1/Relu:activations:0'model/fc2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџД
 model/fc2/BiasAdd/ReadVariableOpReadVariableOp)model_fc2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:
model/fc2/BiasAddBiasAddmodel/fc2/MatMul:product:0(model/fc2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
model/fc2/ReluRelumodel/fc2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџЦ
'model/predictions/MatMul/ReadVariableOpReadVariableOp0model_predictions_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes

:*
dtype0Ѓ
model/predictions/MatMulMatMulmodel/fc2/Relu:activations:0/model/predictions/MatMul/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ*
T0Ф
(model/predictions/BiasAdd/ReadVariableOpReadVariableOp1model_predictions_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:Ќ
model/predictions/BiasAddBiasAdd"model/predictions/MatMul:product:00model/predictions/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ*
T0z
model/predictions/SoftmaxSoftmax"model/predictions/BiasAdd:output:0*'
_output_shapes
:џџџџџџџџџ*
T0Е
IdentityIdentity#model/predictions/Softmax:softmax:0*^model/block1_conv1/BiasAdd/ReadVariableOp)^model/block1_conv1/Conv2D/ReadVariableOp*^model/block1_conv2/BiasAdd/ReadVariableOp)^model/block1_conv2/Conv2D/ReadVariableOp*^model/block2_conv1/BiasAdd/ReadVariableOp)^model/block2_conv1/Conv2D/ReadVariableOp*^model/block2_conv2/BiasAdd/ReadVariableOp)^model/block2_conv2/Conv2D/ReadVariableOp*^model/block3_conv1/BiasAdd/ReadVariableOp)^model/block3_conv1/Conv2D/ReadVariableOp*^model/block3_conv2/BiasAdd/ReadVariableOp)^model/block3_conv2/Conv2D/ReadVariableOp*^model/block3_conv3/BiasAdd/ReadVariableOp)^model/block3_conv3/Conv2D/ReadVariableOp*^model/block4_conv1/BiasAdd/ReadVariableOp)^model/block4_conv1/Conv2D/ReadVariableOp*^model/block4_conv2/BiasAdd/ReadVariableOp)^model/block4_conv2/Conv2D/ReadVariableOp*^model/block4_conv3/BiasAdd/ReadVariableOp)^model/block4_conv3/Conv2D/ReadVariableOp*^model/block5_conv1/BiasAdd/ReadVariableOp)^model/block5_conv1/Conv2D/ReadVariableOp*^model/block5_conv2/BiasAdd/ReadVariableOp)^model/block5_conv2/Conv2D/ReadVariableOp*^model/block5_conv3/BiasAdd/ReadVariableOp)^model/block5_conv3/Conv2D/ReadVariableOp!^model/fc1/BiasAdd/ReadVariableOp ^model/fc1/MatMul/ReadVariableOp!^model/fc2/BiasAdd/ReadVariableOp ^model/fc2/MatMul/ReadVariableOp)^model/predictions/BiasAdd/ReadVariableOp(^model/predictions/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*И
_input_shapesІ
Ѓ:џџџџџџџџџрр:::::::::::::::::::::::::::::::::2T
(model/block1_conv1/Conv2D/ReadVariableOp(model/block1_conv1/Conv2D/ReadVariableOp2T
(model/block5_conv1/Conv2D/ReadVariableOp(model/block5_conv1/Conv2D/ReadVariableOp2T
(model/block4_conv3/Conv2D/ReadVariableOp(model/block4_conv3/Conv2D/ReadVariableOp2T
(model/block1_conv2/Conv2D/ReadVariableOp(model/block1_conv2/Conv2D/ReadVariableOp2T
(model/block5_conv2/Conv2D/ReadVariableOp(model/block5_conv2/Conv2D/ReadVariableOp2V
)model/block3_conv1/BiasAdd/ReadVariableOp)model/block3_conv1/BiasAdd/ReadVariableOp2D
 model/fc1/BiasAdd/ReadVariableOp model/fc1/BiasAdd/ReadVariableOp2V
)model/block4_conv2/BiasAdd/ReadVariableOp)model/block4_conv2/BiasAdd/ReadVariableOp2T
(model/block2_conv1/Conv2D/ReadVariableOp(model/block2_conv1/Conv2D/ReadVariableOp2V
)model/block1_conv2/BiasAdd/ReadVariableOp)model/block1_conv2/BiasAdd/ReadVariableOp2V
)model/block5_conv3/BiasAdd/ReadVariableOp)model/block5_conv3/BiasAdd/ReadVariableOp2T
(model/block5_conv3/Conv2D/ReadVariableOp(model/block5_conv3/Conv2D/ReadVariableOp2T
(model/block2_conv2/Conv2D/ReadVariableOp(model/block2_conv2/Conv2D/ReadVariableOp2R
'model/predictions/MatMul/ReadVariableOp'model/predictions/MatMul/ReadVariableOp2T
(model/predictions/BiasAdd/ReadVariableOp(model/predictions/BiasAdd/ReadVariableOp2T
(model/block3_conv1/Conv2D/ReadVariableOp(model/block3_conv1/Conv2D/ReadVariableOp2V
)model/block4_conv1/BiasAdd/ReadVariableOp)model/block4_conv1/BiasAdd/ReadVariableOp2B
model/fc1/MatMul/ReadVariableOpmodel/fc1/MatMul/ReadVariableOp2V
)model/block1_conv1/BiasAdd/ReadVariableOp)model/block1_conv1/BiasAdd/ReadVariableOp2V
)model/block5_conv2/BiasAdd/ReadVariableOp)model/block5_conv2/BiasAdd/ReadVariableOp2V
)model/block2_conv2/BiasAdd/ReadVariableOp)model/block2_conv2/BiasAdd/ReadVariableOp2V
)model/block3_conv3/BiasAdd/ReadVariableOp)model/block3_conv3/BiasAdd/ReadVariableOp2T
(model/block3_conv2/Conv2D/ReadVariableOp(model/block3_conv2/Conv2D/ReadVariableOp2B
model/fc2/MatMul/ReadVariableOpmodel/fc2/MatMul/ReadVariableOp2T
(model/block4_conv1/Conv2D/ReadVariableOp(model/block4_conv1/Conv2D/ReadVariableOp2T
(model/block3_conv3/Conv2D/ReadVariableOp(model/block3_conv3/Conv2D/ReadVariableOp2V
)model/block5_conv1/BiasAdd/ReadVariableOp)model/block5_conv1/BiasAdd/ReadVariableOp2V
)model/block2_conv1/BiasAdd/ReadVariableOp)model/block2_conv1/BiasAdd/ReadVariableOp2V
)model/block3_conv2/BiasAdd/ReadVariableOp)model/block3_conv2/BiasAdd/ReadVariableOp2T
(model/block4_conv2/Conv2D/ReadVariableOp(model/block4_conv2/Conv2D/ReadVariableOp2V
)model/block4_conv3/BiasAdd/ReadVariableOp)model/block4_conv3/BiasAdd/ReadVariableOp2D
 model/fc2/BiasAdd/ReadVariableOp model/fc2/BiasAdd/ReadVariableOp:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! 
Г
­
,__inference_block2_conv2_layer_call_fn_18667

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_18656*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*
Tin
2*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*,
_gradient_op_typePartitionedCall-18662
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 

b
F__inference_block4_pool_layer_call_and_return_conditional_losses_18859

inputs
identityЂ
MaxPoolMaxPoolinputs*
strides
*
ksize
*
paddingVALID*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ{
IdentityIdentityMaxPool:output:0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:& "
 
_user_specified_nameinputs
А
­
,__inference_block1_conv1_layer_call_fn_18575

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*,
_gradient_op_typePartitionedCall-18570*P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_18564*
Tout
2*/
config_proto

CPU

GPU2 *0J 8
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
р
Ќ
+__inference_predictions_layer_call_fn_19889

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*'
_output_shapes
:џџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-19144*O
fJRH
F__inference_predictions_layer_call_and_return_conditional_losses_19138*
Tout
2*/
config_proto

CPU

GPU2 *0J 8
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 

р
G__inference_block2_conv1_layer_call_and_return_conditional_losses_18631

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*'
_output_shapes
:@*
dtype0Ќ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingSAME*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0*
strides
Ё
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџk
ReluReluBiasAdd:output:0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0І
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
Г
Ё

%__inference_model_layer_call_fn_19318
input_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33(
$statefulpartitionedcall_lambda_const
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1$statefulpartitionedcall_lambda_conststatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33*/
config_proto

CPU

GPU2 *0J 8*-
Tin&
$2"*'
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-19282*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_19281*
Tout
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*И
_input_shapesІ
Ѓ:џџџџџџџџџрр:::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : : : : : : : : :  :! :' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : : : : : : : 
Ї
G
+__inference_block2_pool_layer_call_fn_18684

inputs
identityУ
PartitionedCallPartitionedCallinputs*/
config_proto

CPU

GPU2 *0J 8*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-18681*O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_18675*
Tout
2
IdentityIdentityPartitionedCall:output:0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:& "
 
_user_specified_nameinputs
О
o
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_18969

inputs
identityg
Mean/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ^
IdentityIdentityMean:output:0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:& "
 
_user_specified_nameinputs



#__inference_signature_wrapper_19467
input_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33(
$statefulpartitionedcall_lambda_const
identityЂStatefulPartitionedCallъ

StatefulPartitionedCallStatefulPartitionedCallinput_1$statefulpartitionedcall_lambda_conststatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33*,
_gradient_op_typePartitionedCall-19431*)
f$R"
 __inference__wrapped_model_18550*
Tout
2*/
config_proto

CPU

GPU2 *0J 8*'
_output_shapes
:џџџџџџџџџ*-
Tin&
$2"
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*И
_input_shapesІ
Ѓ:џџџџџџџџџрр:::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! 

р
G__inference_block4_conv1_layer_call_and_return_conditional_losses_18790

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:Ќ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0*
strides
*
paddingSAMEЁ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0k
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџІ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*И
serving_defaultЄ
E
input_1:
serving_default_input_1:0џџџџџџџџџрр?
predictions0
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:То
Э
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer_with_weights-12
layer-18
layer-19
layer-20
layer_with_weights-13
layer-21
layer_with_weights-14
layer-22
layer_with_weights-15
layer-23
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
+М&call_and_return_all_conditional_losses
Н_default_save_signature
О__call__"аХ
_tf_keras_modelЕХ{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"name": "input_1", "class_name": "InputLayer", "config": {"batch_input_shape": [null, 224, 224, 3], "dtype": "float32", "sparse": false, "name": "input_1"}, "inbound_nodes": []}, {"name": "lambda", "class_name": "Lambda", "config": {"name": "lambda", "trainable": false, "dtype": "float32", "function": ["4wAAAAAAAAAAAgAAAAMAAAAfAAAAczIAAAB0AHwBZAE8AGQCfAFrB3IYdAF8AWQCPAB0AnwBZAM8\nAHQDfAFkBDwAiAB8AHwBjgFTACkFTtoHYmFja2VuZNoGbGF5ZXJz2gZtb2RlbHPaBXV0aWxzKQRy\nAQAAAHICAAAAcgMAAAByBAAAACkC2gRhcmdz2gZrd2FyZ3MpAdoIYmFzZV9mdW6pAPqEL2hvbWUv\nc2l0ZWd1aS9hbmFjb25kYTMvZW52cy9jZWNpLW5lc3QtcGFzLXVuLWNoYXQvbGliL3B5dGhvbjMu\nNy9zaXRlLXBhY2thZ2VzL3RlbnNvcmZsb3dfY29yZS9weXRob24va2VyYXMvYXBwbGljYXRpb25z\nL19faW5pdF9fLnB52gd3cmFwcGVyKwAAAHMMAAAAAAEIAQgBCAEIAQgB\n", null, ["preprocess_input"]], "function_type": "lambda", "module": "tensorflow.python.keras.applications", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"name": "block1_conv1", "class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["lambda", 0, 0, {}]]]}, {"name": "block1_conv2", "class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"name": "block1_pool", "class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": false, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"name": "block2_conv1", "class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block1_pool", 0, 0, {}]]]}, {"name": "block2_conv2", "class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"name": "block2_pool", "class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": false, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"name": "block3_conv1", "class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block2_pool", 0, 0, {}]]]}, {"name": "block3_conv2", "class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block3_conv1", 0, 0, {}]]]}, {"name": "block3_conv3", "class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block3_conv2", 0, 0, {}]]]}, {"name": "block3_pool", "class_name": "MaxPooling2D", "config": {"name": "block3_pool", "trainable": false, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "inbound_nodes": [[["block3_conv3", 0, 0, {}]]]}, {"name": "block4_conv1", "class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block3_pool", 0, 0, {}]]]}, {"name": "block4_conv2", "class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block4_conv1", 0, 0, {}]]]}, {"name": "block4_conv3", "class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block4_conv2", 0, 0, {}]]]}, {"name": "block4_pool", "class_name": "MaxPooling2D", "config": {"name": "block4_pool", "trainable": false, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "inbound_nodes": [[["block4_conv3", 0, 0, {}]]]}, {"name": "block5_conv1", "class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block4_pool", 0, 0, {}]]]}, {"name": "block5_conv2", "class_name": "Conv2D", "config": {"name": "block5_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block5_conv1", 0, 0, {}]]]}, {"name": "block5_conv3", "class_name": "Conv2D", "config": {"name": "block5_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block5_conv2", 0, 0, {}]]]}, {"name": "block5_pool", "class_name": "MaxPooling2D", "config": {"name": "block5_pool", "trainable": false, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "inbound_nodes": [[["block5_conv3", 0, 0, {}]]]}, {"name": "global_average_pooling2d", "class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": false, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["block5_pool", 0, 0, {}]]]}, {"name": "fc1", "class_name": "Dense", "config": {"name": "fc1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}, {"name": "fc2", "class_name": "Dense", "config": {"name": "fc2", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["fc1", 0, 0, {}]]]}, {"name": "predictions", "class_name": "Dense", "config": {"name": "predictions", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["fc2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["predictions", 0, 0]]}, "input_spec": null, "activity_regularizer": null, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"name": "input_1", "class_name": "InputLayer", "config": {"batch_input_shape": [null, 224, 224, 3], "dtype": "float32", "sparse": false, "name": "input_1"}, "inbound_nodes": []}, {"name": "lambda", "class_name": "Lambda", "config": {"name": "lambda", "trainable": false, "dtype": "float32", "function": ["4wAAAAAAAAAAAgAAAAMAAAAfAAAAczIAAAB0AHwBZAE8AGQCfAFrB3IYdAF8AWQCPAB0AnwBZAM8\nAHQDfAFkBDwAiAB8AHwBjgFTACkFTtoHYmFja2VuZNoGbGF5ZXJz2gZtb2RlbHPaBXV0aWxzKQRy\nAQAAAHICAAAAcgMAAAByBAAAACkC2gRhcmdz2gZrd2FyZ3MpAdoIYmFzZV9mdW6pAPqEL2hvbWUv\nc2l0ZWd1aS9hbmFjb25kYTMvZW52cy9jZWNpLW5lc3QtcGFzLXVuLWNoYXQvbGliL3B5dGhvbjMu\nNy9zaXRlLXBhY2thZ2VzL3RlbnNvcmZsb3dfY29yZS9weXRob24va2VyYXMvYXBwbGljYXRpb25z\nL19faW5pdF9fLnB52gd3cmFwcGVyKwAAAHMMAAAAAAEIAQgBCAEIAQgB\n", null, ["preprocess_input"]], "function_type": "lambda", "module": "tensorflow.python.keras.applications", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"name": "block1_conv1", "class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["lambda", 0, 0, {}]]]}, {"name": "block1_conv2", "class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"name": "block1_pool", "class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": false, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"name": "block2_conv1", "class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block1_pool", 0, 0, {}]]]}, {"name": "block2_conv2", "class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"name": "block2_pool", "class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": false, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"name": "block3_conv1", "class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block2_pool", 0, 0, {}]]]}, {"name": "block3_conv2", "class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block3_conv1", 0, 0, {}]]]}, {"name": "block3_conv3", "class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block3_conv2", 0, 0, {}]]]}, {"name": "block3_pool", "class_name": "MaxPooling2D", "config": {"name": "block3_pool", "trainable": false, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "inbound_nodes": [[["block3_conv3", 0, 0, {}]]]}, {"name": "block4_conv1", "class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block3_pool", 0, 0, {}]]]}, {"name": "block4_conv2", "class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block4_conv1", 0, 0, {}]]]}, {"name": "block4_conv3", "class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block4_conv2", 0, 0, {}]]]}, {"name": "block4_pool", "class_name": "MaxPooling2D", "config": {"name": "block4_pool", "trainable": false, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "inbound_nodes": [[["block4_conv3", 0, 0, {}]]]}, {"name": "block5_conv1", "class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block4_pool", 0, 0, {}]]]}, {"name": "block5_conv2", "class_name": "Conv2D", "config": {"name": "block5_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block5_conv1", 0, 0, {}]]]}, {"name": "block5_conv3", "class_name": "Conv2D", "config": {"name": "block5_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["block5_conv2", 0, 0, {}]]]}, {"name": "block5_pool", "class_name": "MaxPooling2D", "config": {"name": "block5_pool", "trainable": false, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "inbound_nodes": [[["block5_conv3", 0, 0, {}]]]}, {"name": "global_average_pooling2d", "class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": false, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["block5_pool", 0, 0, {}]]]}, {"name": "fc1", "class_name": "Dense", "config": {"name": "fc1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}, {"name": "fc2", "class_name": "Dense", "config": {"name": "fc2", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["fc1", 0, 0, {}]]]}, {"name": "predictions", "class_name": "Dense", "config": {"name": "predictions", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["fc2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["predictions", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ъ
regularization_losses
 	variables
!trainable_variables
"	keras_api
+П&call_and_return_all_conditional_losses
Р__call__"й
_tf_keras_layerП{"class_name": "InputLayer", "name": "input_1", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 224, 224, 3], "config": {"batch_input_shape": [null, 224, 224, 3], "dtype": "float32", "sparse": false, "name": "input_1"}, "input_spec": null, "activity_regularizer": null}
Ж	
#	arguments
$_variable_dict
%_trainable_weights
&_non_trainable_weights
'_callable_losses
(regularization_losses
)	variables
*trainable_variables
+	keras_api
+С&call_and_return_all_conditional_losses
Т__call__"И
_tf_keras_layer{"class_name": "Lambda", "name": "lambda", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "lambda", "trainable": false, "dtype": "float32", "function": ["4wAAAAAAAAAAAgAAAAMAAAAfAAAAczIAAAB0AHwBZAE8AGQCfAFrB3IYdAF8AWQCPAB0AnwBZAM8\nAHQDfAFkBDwAiAB8AHwBjgFTACkFTtoHYmFja2VuZNoGbGF5ZXJz2gZtb2RlbHPaBXV0aWxzKQRy\nAQAAAHICAAAAcgMAAAByBAAAACkC2gRhcmdz2gZrd2FyZ3MpAdoIYmFzZV9mdW6pAPqEL2hvbWUv\nc2l0ZWd1aS9hbmFjb25kYTMvZW52cy9jZWNpLW5lc3QtcGFzLXVuLWNoYXQvbGliL3B5dGhvbjMu\nNy9zaXRlLXBhY2thZ2VzL3RlbnNvcmZsb3dfY29yZS9weXRob24va2VyYXMvYXBwbGljYXRpb25z\nL19faW5pdF9fLnB52gd3cmFwcGVyKwAAAHMMAAAAAAEIAQgBCAEIAQgB\n", null, ["preprocess_input"]], "function_type": "lambda", "module": "tensorflow.python.keras.applications", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "input_spec": null, "activity_regularizer": null}
Ћ

,kernel
-bias
._callable_losses
/regularization_losses
0	variables
1trainable_variables
2	keras_api
+У&call_and_return_all_conditional_losses
Ф__call__"ю
_tf_keras_layerд{"class_name": "Conv2D", "name": "block1_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block1_conv1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "activity_regularizer": null}
Ќ

3kernel
4bias
5_callable_losses
6regularization_losses
7	variables
8trainable_variables
9	keras_api
+Х&call_and_return_all_conditional_losses
Ц__call__"я
_tf_keras_layerе{"class_name": "Conv2D", "name": "block1_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block1_conv2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "activity_regularizer": null}
­
:_callable_losses
;regularization_losses
<	variables
=trainable_variables
>	keras_api
+Ч&call_and_return_all_conditional_losses
Ш__call__"
_tf_keras_layerь{"class_name": "MaxPooling2D", "name": "block1_pool", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block1_pool", "trainable": false, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "activity_regularizer": null}
­

?kernel
@bias
A_callable_losses
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
+Щ&call_and_return_all_conditional_losses
Ъ__call__"№
_tf_keras_layerж{"class_name": "Conv2D", "name": "block2_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "activity_regularizer": null}
Ў

Fkernel
Gbias
H_callable_losses
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
+Ы&call_and_return_all_conditional_losses
Ь__call__"ё
_tf_keras_layerз{"class_name": "Conv2D", "name": "block2_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "activity_regularizer": null}
­
M_callable_losses
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
+Э&call_and_return_all_conditional_losses
Ю__call__"
_tf_keras_layerь{"class_name": "MaxPooling2D", "name": "block2_pool", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block2_pool", "trainable": false, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "activity_regularizer": null}
Ў

Rkernel
Sbias
T_callable_losses
Uregularization_losses
V	variables
Wtrainable_variables
X	keras_api
+Я&call_and_return_all_conditional_losses
а__call__"ё
_tf_keras_layerз{"class_name": "Conv2D", "name": "block3_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block3_conv1", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "activity_regularizer": null}
Ў

Ykernel
Zbias
[_callable_losses
\regularization_losses
]	variables
^trainable_variables
_	keras_api
+б&call_and_return_all_conditional_losses
в__call__"ё
_tf_keras_layerз{"class_name": "Conv2D", "name": "block3_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block3_conv2", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}, "activity_regularizer": null}
Ў

`kernel
abias
b_callable_losses
cregularization_losses
d	variables
etrainable_variables
f	keras_api
+г&call_and_return_all_conditional_losses
д__call__"ё
_tf_keras_layerз{"class_name": "Conv2D", "name": "block3_conv3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block3_conv3", "trainable": false, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}, "activity_regularizer": null}
­
g_callable_losses
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
+е&call_and_return_all_conditional_losses
ж__call__"
_tf_keras_layerь{"class_name": "MaxPooling2D", "name": "block3_pool", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block3_pool", "trainable": false, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "activity_regularizer": null}
Ў

lkernel
mbias
n_callable_losses
oregularization_losses
p	variables
qtrainable_variables
r	keras_api
+з&call_and_return_all_conditional_losses
и__call__"ё
_tf_keras_layerз{"class_name": "Conv2D", "name": "block4_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block4_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}, "activity_regularizer": null}
Ў

skernel
tbias
u_callable_losses
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
+й&call_and_return_all_conditional_losses
к__call__"ё
_tf_keras_layerз{"class_name": "Conv2D", "name": "block4_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block4_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}, "activity_regularizer": null}
Џ

zkernel
{bias
|_callable_losses
}regularization_losses
~	variables
trainable_variables
	keras_api
+л&call_and_return_all_conditional_losses
м__call__"ё
_tf_keras_layerз{"class_name": "Conv2D", "name": "block4_conv3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block4_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}, "activity_regularizer": null}
В
_callable_losses
regularization_losses
	variables
trainable_variables
	keras_api
+н&call_and_return_all_conditional_losses
о__call__"
_tf_keras_layerь{"class_name": "MaxPooling2D", "name": "block4_pool", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block4_pool", "trainable": false, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "activity_regularizer": null}
Е
kernel
	bias
_callable_losses
regularization_losses
	variables
trainable_variables
	keras_api
+п&call_and_return_all_conditional_losses
р__call__"ё
_tf_keras_layerз{"class_name": "Conv2D", "name": "block5_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block5_conv1", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}, "activity_regularizer": null}
Е
kernel
	bias
_callable_losses
regularization_losses
	variables
trainable_variables
	keras_api
+с&call_and_return_all_conditional_losses
т__call__"ё
_tf_keras_layerз{"class_name": "Conv2D", "name": "block5_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block5_conv2", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}, "activity_regularizer": null}
Е
kernel
	bias
_callable_losses
regularization_losses
	variables
trainable_variables
	keras_api
+у&call_and_return_all_conditional_losses
ф__call__"ё
_tf_keras_layerз{"class_name": "Conv2D", "name": "block5_conv3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block5_conv3", "trainable": false, "dtype": "float32", "filters": 512, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}, "activity_regularizer": null}
В
_callable_losses
regularization_losses
	variables
trainable_variables
	keras_api
+х&call_and_return_all_conditional_losses
ц__call__"
_tf_keras_layerь{"class_name": "MaxPooling2D", "name": "block5_pool", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "block5_pool", "trainable": false, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "activity_regularizer": null}

 _callable_losses
Ёregularization_losses
Ђ	variables
Ѓtrainable_variables
Є	keras_api
+ч&call_and_return_all_conditional_losses
ш__call__"ю
_tf_keras_layerд{"class_name": "GlobalAveragePooling2D", "name": "global_average_pooling2d", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "global_average_pooling2d", "trainable": false, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "activity_regularizer": null}
Ј
Ѕkernel
	Іbias
Ї_callable_losses
Јregularization_losses
Љ	variables
Њtrainable_variables
Ћ	keras_api
+щ&call_and_return_all_conditional_losses
ъ__call__"ф
_tf_keras_layerЪ{"class_name": "Dense", "name": "fc1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "fc1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "activity_regularizer": null}
Ї
Ќkernel
	­bias
Ў_callable_losses
Џregularization_losses
А	variables
Бtrainable_variables
В	keras_api
+ы&call_and_return_all_conditional_losses
ь__call__"у
_tf_keras_layerЩ{"class_name": "Dense", "name": "fc2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "fc2", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "activity_regularizer": null}
И
Гkernel
	Дbias
Е_callable_losses
Жregularization_losses
З	variables
Иtrainable_variables
Й	keras_api
+э&call_and_return_all_conditional_losses
ю__call__"є
_tf_keras_layerк{"class_name": "Dense", "name": "predictions", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "predictions", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "activity_regularizer": null}
м
	Кiter
Лbeta_1
Мbeta_2

Нdecay
Оlearning_rate	ЅmА	ІmБ	ЌmВ	­mГ	ГmД	ДmЕ	ЅvЖ	ІvЗ	ЌvИ	­vЙ	ГvК	ДvЛ"
	optimizer
 "
trackable_list_wrapper
Ђ
,0
-1
32
43
?4
@5
F6
G7
R8
S9
Y10
Z11
`12
a13
l14
m15
s16
t17
z18
{19
20
21
22
23
24
25
Ѕ26
І27
Ќ28
­29
Г30
Д31"
trackable_list_wrapper
P
Ѕ0
І1
Ќ2
­3
Г4
Д5"
trackable_list_wrapper
П
 Пlayer_regularization_losses
Рlayers
regularization_losses
	variables
Сmetrics
trainable_variables
Тnon_trainable_variables
О__call__
Н_default_save_signature
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
-
яserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
 Уlayer_regularization_losses
Фlayers
regularization_losses
 	variables
Хmetrics
!trainable_variables
Цnon_trainable_variables
Р__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
 Чlayer_regularization_losses
Шlayers
(regularization_losses
)	variables
Щmetrics
*trainable_variables
Ъnon_trainable_variables
Т__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
-:+@2block1_conv1/kernel
:@2block1_conv1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
 Ыlayer_regularization_losses
Ьlayers
/regularization_losses
0	variables
Эmetrics
1trainable_variables
Юnon_trainable_variables
Ф__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
-:+@@2block1_conv2/kernel
:@2block1_conv2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
 Яlayer_regularization_losses
аlayers
6regularization_losses
7	variables
бmetrics
8trainable_variables
вnon_trainable_variables
Ц__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
 гlayer_regularization_losses
дlayers
;regularization_losses
<	variables
еmetrics
=trainable_variables
жnon_trainable_variables
Ш__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
.:,@2block2_conv1/kernel
 :2block2_conv1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
 зlayer_regularization_losses
иlayers
Bregularization_losses
C	variables
йmetrics
Dtrainable_variables
кnon_trainable_variables
Ъ__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
/:-2block2_conv2/kernel
 :2block2_conv2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
 лlayer_regularization_losses
мlayers
Iregularization_losses
J	variables
нmetrics
Ktrainable_variables
оnon_trainable_variables
Ь__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
 пlayer_regularization_losses
рlayers
Nregularization_losses
O	variables
сmetrics
Ptrainable_variables
тnon_trainable_variables
Ю__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
/:-2block3_conv1/kernel
 :2block3_conv1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
 уlayer_regularization_losses
фlayers
Uregularization_losses
V	variables
хmetrics
Wtrainable_variables
цnon_trainable_variables
а__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
/:-2block3_conv2/kernel
 :2block3_conv2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
 чlayer_regularization_losses
шlayers
\regularization_losses
]	variables
щmetrics
^trainable_variables
ъnon_trainable_variables
в__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
/:-2block3_conv3/kernel
 :2block3_conv3/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
 ыlayer_regularization_losses
ьlayers
cregularization_losses
d	variables
эmetrics
etrainable_variables
юnon_trainable_variables
д__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
 яlayer_regularization_losses
№layers
hregularization_losses
i	variables
ёmetrics
jtrainable_variables
ђnon_trainable_variables
ж__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
/:-2block4_conv1/kernel
 :2block4_conv1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
 ѓlayer_regularization_losses
єlayers
oregularization_losses
p	variables
ѕmetrics
qtrainable_variables
іnon_trainable_variables
и__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
/:-2block4_conv2/kernel
 :2block4_conv2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
 їlayer_regularization_losses
јlayers
vregularization_losses
w	variables
љmetrics
xtrainable_variables
њnon_trainable_variables
к__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
/:-2block4_conv3/kernel
 :2block4_conv3/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ё
 ћlayer_regularization_losses
ќlayers
}regularization_losses
~	variables
§metrics
trainable_variables
ўnon_trainable_variables
м__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
 џlayer_regularization_losses
layers
regularization_losses
	variables
metrics
trainable_variables
non_trainable_variables
о__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
/:-2block5_conv1/kernel
 :2block5_conv1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
 layer_regularization_losses
layers
regularization_losses
	variables
metrics
trainable_variables
non_trainable_variables
р__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
/:-2block5_conv2/kernel
 :2block5_conv2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
 layer_regularization_losses
layers
regularization_losses
	variables
metrics
trainable_variables
non_trainable_variables
т__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
/:-2block5_conv3/kernel
 :2block5_conv3/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
 layer_regularization_losses
layers
regularization_losses
	variables
metrics
trainable_variables
non_trainable_variables
ф__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
 layer_regularization_losses
layers
regularization_losses
	variables
metrics
trainable_variables
non_trainable_variables
ц__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
 layer_regularization_losses
layers
Ёregularization_losses
Ђ	variables
metrics
Ѓtrainable_variables
non_trainable_variables
ш__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses"
_generic_user_object
:
2
fc1/kernel
:2fc1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ѕ0
І1"
trackable_list_wrapper
0
Ѕ0
І1"
trackable_list_wrapper
Є
 layer_regularization_losses
layers
Јregularization_losses
Љ	variables
metrics
Њtrainable_variables
non_trainable_variables
ъ__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
:	2
fc2/kernel
:2fc2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ќ0
­1"
trackable_list_wrapper
0
Ќ0
­1"
trackable_list_wrapper
Є
 layer_regularization_losses
layers
Џregularization_losses
А	variables
metrics
Бtrainable_variables
non_trainable_variables
ь__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
$:"2predictions/kernel
:2predictions/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Г0
Д1"
trackable_list_wrapper
0
Г0
Д1"
trackable_list_wrapper
Є
 layer_regularization_losses
 layers
Жregularization_losses
З	variables
Ёmetrics
Иtrainable_variables
Ђnon_trainable_variables
ю__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
:	 (2training/Adam/iter
: (2training/Adam/beta_1
: (2training/Adam/beta_2
: (2training/Adam/decay
%:# (2training/Adam/learning_rate
 "
trackable_list_wrapper
ж
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23"
trackable_list_wrapper
(
Ѓ0"
trackable_list_wrapper
ь
,0
-1
32
43
?4
@5
F6
G7
R8
S9
Y10
Z11
`12
a13
l14
m15
s16
t17
z18
{19
20
21
22
23
24
25"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ф

Єtotal

Ѕcount
І
_fn_kwargs
Ї_updates
Јregularization_losses
Љ	variables
Њtrainable_variables
Ћ	keras_api
+№&call_and_return_all_conditional_losses
ё__call__"
_tf_keras_layer§{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}, "input_spec": null, "activity_regularizer": null}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Є0
Ѕ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
 Ќlayer_regularization_losses
­layers
Јregularization_losses
Љ	variables
Ўmetrics
Њtrainable_variables
Џnon_trainable_variables
ё__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Є0
Ѕ1"
trackable_list_wrapper
,:*
2training/Adam/fc1/kernel/m
%:#2training/Adam/fc1/bias/m
+:)	2training/Adam/fc2/kernel/m
$:"2training/Adam/fc2/bias/m
2:02"training/Adam/predictions/kernel/m
,:*2 training/Adam/predictions/bias/m
,:*
2training/Adam/fc1/kernel/v
%:#2training/Adam/fc1/bias/v
+:)	2training/Adam/fc2/kernel/v
$:"2training/Adam/fc2/bias/v
2:02"training/Adam/predictions/kernel/v
,:*2 training/Adam/predictions/bias/v
Ю2Ы
@__inference_model_layer_call_and_return_conditional_losses_19598
@__inference_model_layer_call_and_return_conditional_losses_19156
@__inference_model_layer_call_and_return_conditional_losses_19218
@__inference_model_layer_call_and_return_conditional_losses_19727Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ш2х
 __inference__wrapped_model_18550Р
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *0Ђ-
+(
input_1џџџџџџџџџрр
т2п
%__inference_model_layer_call_fn_19419
%__inference_model_layer_call_fn_19765
%__inference_model_layer_call_fn_19318
%__inference_model_layer_call_fn_19803Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
Ь2Щ
A__inference_lambda_layer_call_and_return_conditional_losses_19823
A__inference_lambda_layer_call_and_return_conditional_losses_19813Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
&__inference_lambda_layer_call_fn_19829
&__inference_lambda_layer_call_fn_19835Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
І2Ѓ
G__inference_block1_conv1_layer_call_and_return_conditional_losses_18564з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
,__inference_block1_conv1_layer_call_fn_18575з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
І2Ѓ
G__inference_block1_conv2_layer_call_and_return_conditional_losses_18589з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
2
,__inference_block1_conv2_layer_call_fn_18600з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Ў2Ћ
F__inference_block1_pool_layer_call_and_return_conditional_losses_18608р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
+__inference_block1_pool_layer_call_fn_18617р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
І2Ѓ
G__inference_block2_conv1_layer_call_and_return_conditional_losses_18631з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
2
,__inference_block2_conv1_layer_call_fn_18642з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Ї2Є
G__inference_block2_conv2_layer_call_and_return_conditional_losses_18656и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
,__inference_block2_conv2_layer_call_fn_18667и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ў2Ћ
F__inference_block2_pool_layer_call_and_return_conditional_losses_18675р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
+__inference_block2_pool_layer_call_fn_18684р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ї2Є
G__inference_block3_conv1_layer_call_and_return_conditional_losses_18698и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
,__inference_block3_conv1_layer_call_fn_18709и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ї2Є
G__inference_block3_conv2_layer_call_and_return_conditional_losses_18723и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
,__inference_block3_conv2_layer_call_fn_18734и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ї2Є
G__inference_block3_conv3_layer_call_and_return_conditional_losses_18748и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
,__inference_block3_conv3_layer_call_fn_18759и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ў2Ћ
F__inference_block3_pool_layer_call_and_return_conditional_losses_18767р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
+__inference_block3_pool_layer_call_fn_18776р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ї2Є
G__inference_block4_conv1_layer_call_and_return_conditional_losses_18790и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
,__inference_block4_conv1_layer_call_fn_18801и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ї2Є
G__inference_block4_conv2_layer_call_and_return_conditional_losses_18815и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
,__inference_block4_conv2_layer_call_fn_18826и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ї2Є
G__inference_block4_conv3_layer_call_and_return_conditional_losses_18840и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
,__inference_block4_conv3_layer_call_fn_18851и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ў2Ћ
F__inference_block4_pool_layer_call_and_return_conditional_losses_18859р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
+__inference_block4_pool_layer_call_fn_18868р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ї2Є
G__inference_block5_conv1_layer_call_and_return_conditional_losses_18882и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
,__inference_block5_conv1_layer_call_fn_18893и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ї2Є
G__inference_block5_conv2_layer_call_and_return_conditional_losses_18907и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
,__inference_block5_conv2_layer_call_fn_18918и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ї2Є
G__inference_block5_conv3_layer_call_and_return_conditional_losses_18932и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
,__inference_block5_conv3_layer_call_fn_18943и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ў2Ћ
F__inference_block5_pool_layer_call_and_return_conditional_losses_18951р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
+__inference_block5_pool_layer_call_fn_18960р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Л2И
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_18969р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 2
8__inference_global_average_pooling2d_layer_call_fn_18978р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
ш2х
>__inference_fc1_layer_call_and_return_conditional_losses_19846Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Э2Ъ
#__inference_fc1_layer_call_fn_19853Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ш2х
>__inference_fc2_layer_call_and_return_conditional_losses_19864Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Э2Ъ
#__inference_fc2_layer_call_fn_19871Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_predictions_layer_call_and_return_conditional_losses_19882Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_predictions_layer_call_fn_19889Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2B0
#__inference_signature_wrapper_19467input_1
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
	J
ConstЗ
%__inference_model_layer_call_fn_19803.,-34?@FGRSYZ`almstz{ЅІЌ­ГДђAЂ>
7Ђ4
*'
inputsџџџџџџџџџрр
p 

 
Њ "џџџџџџџџџм
G__inference_block1_conv1_layer_call_and_return_conditional_losses_18564,-IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 щ
F__inference_block4_pool_layer_call_and_return_conditional_losses_18859RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 З
%__inference_model_layer_call_fn_19765.,-34?@FGRSYZ`almstz{ЅІЌ­ГДђAЂ>
7Ђ4
*'
inputsџџџџџџџџџрр
p

 
Њ "џџџџџџџџџЖ
,__inference_block4_conv3_layer_call_fn_18851z{JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџЖ
,__inference_block4_conv1_layer_call_fn_18801lmJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџИ
%__inference_model_layer_call_fn_19318.,-34?@FGRSYZ`almstz{ЅІЌ­ГДђBЂ?
8Ђ5
+(
input_1џџџџџџџџџрр
p

 
Њ "џџџџџџџџџГ
8__inference_global_average_pooling2d_layer_call_fn_18978wRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "!џџџџџџџџџџџџџџџџџџЁ
>__inference_fc2_layer_call_and_return_conditional_losses_19864_Ќ­0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 п
@__inference_model_layer_call_and_return_conditional_losses_19598.ђ,-34?@FGRSYZ`almstz{ЅІЌ­ГДAЂ>
7Ђ4
*'
inputsџџџџџџџџџрр
p

 
Њ "%Ђ"

0џџџџџџџџџ
 о
G__inference_block3_conv2_layer_call_and_return_conditional_losses_18723YZJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Н
A__inference_lambda_layer_call_and_return_conditional_losses_19813xђAЂ>
7Ђ4
*'
inputsџџџџџџџџџрр

 
p
Њ "/Ђ,
%"
0џџџџџџџџџрр
 о
G__inference_block3_conv1_layer_call_and_return_conditional_losses_18698RSJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 о
G__inference_block2_conv2_layer_call_and_return_conditional_losses_18656FGJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 р
@__inference_model_layer_call_and_return_conditional_losses_19156.ђ,-34?@FGRSYZ`almstz{ЅІЌ­ГДBЂ?
8Ђ5
+(
input_1џџџџџџџџџрр
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Ј
F__inference_predictions_layer_call_and_return_conditional_losses_19882^ГД/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 Н
A__inference_lambda_layer_call_and_return_conditional_losses_19823xђAЂ>
7Ђ4
*'
inputsџџџџџџџџџрр

 
p 
Њ "/Ђ,
%"
0џџџџџџџџџрр
 И
,__inference_block5_conv2_layer_call_fn_18918JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџм
G__inference_block1_conv2_layer_call_and_return_conditional_losses_1858934IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 о
G__inference_block4_conv1_layer_call_and_return_conditional_losses_18790lmJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 y
#__inference_fc2_layer_call_fn_19871RЌ­0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџо
G__inference_block4_conv2_layer_call_and_return_conditional_losses_18815stJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 р
@__inference_model_layer_call_and_return_conditional_losses_19218.ђ,-34?@FGRSYZ`almstz{ЅІЌ­ГДBЂ?
8Ђ5
+(
input_1џџџџџџџџџрр
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 п
@__inference_model_layer_call_and_return_conditional_losses_19727.ђ,-34?@FGRSYZ`almstz{ЅІЌ­ГДAЂ>
7Ђ4
*'
inputsџџџџџџџџџрр
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 щ
F__inference_block1_pool_layer_call_and_return_conditional_losses_18608RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 И
%__inference_model_layer_call_fn_19419.,-34?@FGRSYZ`almstz{ЅІЌ­ГДђBЂ?
8Ђ5
+(
input_1џџџџџџџџџрр
p 

 
Њ "џџџџџџџџџр
G__inference_block5_conv1_layer_call_and_return_conditional_losses_18882JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 р
G__inference_block5_conv2_layer_call_and_return_conditional_losses_18907JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 о
G__inference_block3_conv3_layer_call_and_return_conditional_losses_18748`aJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 С
+__inference_block4_pool_layer_call_fn_18868RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџо
G__inference_block4_conv3_layer_call_and_return_conditional_losses_18840z{JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 щ
F__inference_block3_pool_layer_call_and_return_conditional_losses_18767RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Д
,__inference_block1_conv1_layer_call_fn_18575,-IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@Ж
,__inference_block2_conv2_layer_call_fn_18667FGJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџЖ
,__inference_block3_conv3_layer_call_fn_18759`aJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџС
+__inference_block2_pool_layer_call_fn_18684RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџр
G__inference_block5_conv3_layer_call_and_return_conditional_losses_18932JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ж
,__inference_block3_conv1_layer_call_fn_18709RSJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџщ
F__inference_block5_pool_layer_call_and_return_conditional_losses_18951RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
&__inference_lambda_layer_call_fn_19829kђAЂ>
7Ђ4
*'
inputsџџџџџџџџџрр

 
p
Њ ""џџџџџџџџџрр
&__inference_lambda_layer_call_fn_19835kђAЂ>
7Ђ4
*'
inputsџџџџџџџџџрр

 
p 
Њ ""џџџџџџџџџррк
#__inference_signature_wrapper_19467В.,-34?@FGRSYZ`almstz{ЅІЌ­ГДђEЂB
Ђ 
;Њ8
6
input_1+(
input_1џџџџџџџџџрр"9Њ6
4
predictions%"
predictionsџџџџџџџџџЖ
,__inference_block4_conv2_layer_call_fn_18826stJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџИ
,__inference_block5_conv3_layer_call_fn_18943JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџД
,__inference_block1_conv2_layer_call_fn_1860034IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
+__inference_predictions_layer_call_fn_19889QГД/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџz
#__inference_fc1_layer_call_fn_19853SЅІ0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџС
+__inference_block5_pool_layer_call_fn_18960RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџм
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_18969RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџџџџџџџџџџ
 щ
F__inference_block2_pool_layer_call_and_return_conditional_losses_18675RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ь
 __inference__wrapped_model_18550Ї.ђ,-34?@FGRSYZ`almstz{ЅІЌ­ГД:Ђ7
0Ђ-
+(
input_1џџџџџџџџџрр
Њ "9Њ6
4
predictions%"
predictionsџџџџџџџџџИ
,__inference_block5_conv1_layer_call_fn_18893JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџС
+__inference_block1_pool_layer_call_fn_18617RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџС
+__inference_block3_pool_layer_call_fn_18776RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџЂ
>__inference_fc1_layer_call_and_return_conditional_losses_19846`ЅІ0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 н
G__inference_block2_conv1_layer_call_and_return_conditional_losses_18631?@IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Е
,__inference_block2_conv1_layer_call_fn_18642?@IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџЖ
,__inference_block3_conv2_layer_call_fn_18734YZJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ