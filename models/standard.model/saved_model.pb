��
� � 
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements(
handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.10.12v2.10.0-76-gfdfc646704c8��
�
Adam/conv_lstm2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:4**
shared_nameAdam/conv_lstm2d_3/bias/v
�
-Adam/conv_lstm2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_lstm2d_3/bias/v*
_output_shapes
:4*
dtype0
�
%Adam/conv_lstm2d_3/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:4*6
shared_name'%Adam/conv_lstm2d_3/recurrent_kernel/v
�
9Adam/conv_lstm2d_3/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp%Adam/conv_lstm2d_3/recurrent_kernel/v*&
_output_shapes
:4*
dtype0
�
Adam/conv_lstm2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�4*,
shared_nameAdam/conv_lstm2d_3/kernel/v
�
/Adam/conv_lstm2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_lstm2d_3/kernel/v*'
_output_shapes
:�4*
dtype0
�
Adam/dense_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_11/bias/v
y
(Adam/dense_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*'
shared_nameAdam/dense_11/kernel/v
�
*Adam/dense_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/v*
_output_shapes

:2*
dtype0
�
Adam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/dense_10/bias/v
y
(Adam/dense_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/v*
_output_shapes
:2*
dtype0
�
Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�2*'
shared_nameAdam/dense_10/kernel/v
�
*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v*
_output_shapes
:	�2*
dtype0

Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/dense_9/bias/v
x
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	'�*&
shared_nameAdam/dense_9/kernel/v
�
)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes
:	'�*
dtype0
�
Adam/conv_lstm2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:4**
shared_nameAdam/conv_lstm2d_3/bias/m
�
-Adam/conv_lstm2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_lstm2d_3/bias/m*
_output_shapes
:4*
dtype0
�
%Adam/conv_lstm2d_3/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:4*6
shared_name'%Adam/conv_lstm2d_3/recurrent_kernel/m
�
9Adam/conv_lstm2d_3/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp%Adam/conv_lstm2d_3/recurrent_kernel/m*&
_output_shapes
:4*
dtype0
�
Adam/conv_lstm2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�4*,
shared_nameAdam/conv_lstm2d_3/kernel/m
�
/Adam/conv_lstm2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_lstm2d_3/kernel/m*'
_output_shapes
:�4*
dtype0
�
Adam/dense_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_11/bias/m
y
(Adam/dense_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*'
shared_nameAdam/dense_11/kernel/m
�
*Adam/dense_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/m*
_output_shapes

:2*
dtype0
�
Adam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/dense_10/bias/m
y
(Adam/dense_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/m*
_output_shapes
:2*
dtype0
�
Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�2*'
shared_nameAdam/dense_10/kernel/m
�
*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m*
_output_shapes
:	�2*
dtype0

Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/dense_9/bias/m
x
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	'�*&
shared_nameAdam/dense_9/kernel/m
�
)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes
:	'�*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
|
conv_lstm2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:4*#
shared_nameconv_lstm2d_3/bias
u
&conv_lstm2d_3/bias/Read/ReadVariableOpReadVariableOpconv_lstm2d_3/bias*
_output_shapes
:4*
dtype0
�
conv_lstm2d_3/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:4*/
shared_name conv_lstm2d_3/recurrent_kernel
�
2conv_lstm2d_3/recurrent_kernel/Read/ReadVariableOpReadVariableOpconv_lstm2d_3/recurrent_kernel*&
_output_shapes
:4*
dtype0
�
conv_lstm2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�4*%
shared_nameconv_lstm2d_3/kernel
�
(conv_lstm2d_3/kernel/Read/ReadVariableOpReadVariableOpconv_lstm2d_3/kernel*'
_output_shapes
:�4*
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
:*
dtype0
z
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2* 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

:2*
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
:2*
dtype0
{
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�2* 
shared_namedense_10/kernel
t
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes
:	�2*
dtype0
q
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_9/bias
j
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes	
:�*
dtype0
y
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	'�*
shared_namedense_9/kernel
r
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes
:	'�*
dtype0
�
serving_default_reshape_3_inputPlaceholder*,
_output_shapes
:����������*
dtype0*!
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_reshape_3_inputconv_lstm2d_3/kernelconv_lstm2d_3/recurrent_kernelconv_lstm2d_3/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference_signature_wrapper_70354

NoOpNoOp
�a
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�a
value�aB�a B�`
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-1
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer-9
layer_with_weights-3
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses
!cell
"
state_spec*
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)_random_generator* 
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses* 
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6_random_generator* 
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses* 
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

Ckernel
Dbias*
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses
K_random_generator* 
�
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses

Rkernel
Sbias*
�
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses
Z_random_generator* 
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses

akernel
bbias*
C
c0
d1
e2
C3
D4
R5
S6
a7
b8*
C
c0
d1
e2
C3
D4
R5
S6
a7
b8*
* 
�
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
ktrace_0
ltrace_1
mtrace_2
ntrace_3* 
6
otrace_0
ptrace_1
qtrace_2
rtrace_3* 
* 
�
siter

tbeta_1

ubeta_2
	vdecay
wlearning_rateCm�Dm�Rm�Sm�am�bm�cm�dm�em�Cv�Dv�Rv�Sv�av�bv�cv�dv�ev�*

xserving_default* 
* 
* 
* 
�
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

~trace_0* 

trace_0* 

c0
d1
e2*

c0
d1
e2*
* 
�
�states
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator

ckernel
drecurrent_kernel
ebias*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

C0
D1*

C0
D1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_9/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_9/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

R0
S1*

R0
S1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_10/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

a0
b1*

a0
b1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_11/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEconv_lstm2d_3/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEconv_lstm2d_3/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEconv_lstm2d_3/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 
R
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
10*

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

!0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

c0
d1
e2*

c0
d1
e2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
�{
VARIABLE_VALUEAdam/dense_9/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_9/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_10/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_10/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_11/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_11/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/conv_lstm2d_3/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE%Adam/conv_lstm2d_3/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/conv_lstm2d_3/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUEAdam/dense_9/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_9/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_10/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_10/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_11/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_11/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/conv_lstm2d_3/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE%Adam/conv_lstm2d_3/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/conv_lstm2d_3/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOp(conv_lstm2d_3/kernel/Read/ReadVariableOp2conv_lstm2d_3/recurrent_kernel/Read/ReadVariableOp&conv_lstm2d_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp*Adam/dense_10/kernel/m/Read/ReadVariableOp(Adam/dense_10/bias/m/Read/ReadVariableOp*Adam/dense_11/kernel/m/Read/ReadVariableOp(Adam/dense_11/bias/m/Read/ReadVariableOp/Adam/conv_lstm2d_3/kernel/m/Read/ReadVariableOp9Adam/conv_lstm2d_3/recurrent_kernel/m/Read/ReadVariableOp-Adam/conv_lstm2d_3/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOp*Adam/dense_10/kernel/v/Read/ReadVariableOp(Adam/dense_10/bias/v/Read/ReadVariableOp*Adam/dense_11/kernel/v/Read/ReadVariableOp(Adam/dense_11/bias/v/Read/ReadVariableOp/Adam/conv_lstm2d_3/kernel/v/Read/ReadVariableOp9Adam/conv_lstm2d_3/recurrent_kernel/v/Read/ReadVariableOp-Adam/conv_lstm2d_3/bias/v/Read/ReadVariableOpConst*1
Tin*
(2&	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference__traced_save_72418
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/biasconv_lstm2d_3/kernelconv_lstm2d_3/recurrent_kernelconv_lstm2d_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_9/kernel/mAdam/dense_9/bias/mAdam/dense_10/kernel/mAdam/dense_10/bias/mAdam/dense_11/kernel/mAdam/dense_11/bias/mAdam/conv_lstm2d_3/kernel/m%Adam/conv_lstm2d_3/recurrent_kernel/mAdam/conv_lstm2d_3/bias/mAdam/dense_9/kernel/vAdam/dense_9/bias/vAdam/dense_10/kernel/vAdam/dense_10/bias/vAdam/dense_11/kernel/vAdam/dense_11/bias/vAdam/conv_lstm2d_3/kernel/v%Adam/conv_lstm2d_3/recurrent_kernel/vAdam/conv_lstm2d_3/bias/v*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_restore_72536��
�"
�
while_body_69320
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
while_69344_0:�4'
while_69346_0:4
while_69348_0:4
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
while_69344:�4%
while_69346:4
while_69348:4��while/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����        �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:����������*
element_dtype0�
while/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_69344_0while_69346_0while_69348_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:���������:���������:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_conv_lstm_cell_3_layer_call_and_return_conditional_losses_69266r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0&while/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity&while/StatefulPartitionedCall:output:1^while/NoOp*
T0*/
_output_shapes
:����������
while/Identity_5Identity&while/StatefulPartitionedCall:output:2^while/NoOp*
T0*/
_output_shapes
:���������l

while/NoOpNoOp^while/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
while_69344while_69344_0"
while_69346while_69346_0"
while_69348while_69348_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :���������:���������: : : : : 2>
while/StatefulPartitionedCallwhile/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
�
-__inference_conv_lstm2d_3_layer_call_fn_71018

inputs"
unknown:�4#
	unknown_0:4
	unknown_1:4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_70140w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:����������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�[
�
while_body_71339
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0@
%while_split_readvariableop_resource_0:�4A
'while_split_1_readvariableop_resource_0:45
'while_split_2_readvariableop_resource_0:4
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor>
#while_split_readvariableop_resource:�4?
%while_split_1_readvariableop_resource:43
%while_split_2_readvariableop_resource:4��while/split/ReadVariableOp�while/split_1/ReadVariableOp�while/split_2/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����        �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:����������*
element_dtype0W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:�4*
dtype0�
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*`
_output_shapesN
L:�:�:�:�*
	num_splitY
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
:4*
dtype0�
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes
:4*
dtype0�
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split�
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*/
_output_shapes
:����������
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*/
_output_shapes
:����������
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*/
_output_shapes
:����������
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*/
_output_shapes
:����������
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:���������P
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>R
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?o
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*/
_output_shapes
:���������u
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*/
_output_shapes
:���������b
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������Z
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:����������
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:���������R
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��L>R
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*/
_output_shapes
:���������w
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*/
_output_shapes
:���������d
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:���������\
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:���������|
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:����������
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:���������]

while/ReluReluwhile/add_4:z:0*
T0*/
_output_shapes
:���������
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*/
_output_shapes
:���������p
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:����������
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:���������R
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *��L>R
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*/
_output_shapes
:���������w
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*/
_output_shapes
:���������d
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:���������\
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:���������_
while/Relu_1Reluwhile/add_5:z:0*
T0*/
_output_shapes
:����������
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*/
_output_shapes
:���������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_5:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: O
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: [
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: t
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*/
_output_shapes
:���������t
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*/
_output_shapes
:����������

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :���������:���������: : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
c
*__inference_dropout_13_layer_call_fn_71961

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_13_layer_call_and_return_conditional_losses_69877w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
��
�	
 __inference__wrapped_model_68972
reshape_3_inputS
8sequential_3_conv_lstm2d_3_split_readvariableop_resource:�4T
:sequential_3_conv_lstm2d_3_split_1_readvariableop_resource:4H
:sequential_3_conv_lstm2d_3_split_2_readvariableop_resource:4F
3sequential_3_dense_9_matmul_readvariableop_resource:	'�C
4sequential_3_dense_9_biasadd_readvariableop_resource:	�G
4sequential_3_dense_10_matmul_readvariableop_resource:	�2C
5sequential_3_dense_10_biasadd_readvariableop_resource:2F
4sequential_3_dense_11_matmul_readvariableop_resource:2C
5sequential_3_dense_11_biasadd_readvariableop_resource:
identity��/sequential_3/conv_lstm2d_3/split/ReadVariableOp�1sequential_3/conv_lstm2d_3/split_1/ReadVariableOp�1sequential_3/conv_lstm2d_3/split_2/ReadVariableOp� sequential_3/conv_lstm2d_3/while�,sequential_3/dense_10/BiasAdd/ReadVariableOp�+sequential_3/dense_10/MatMul/ReadVariableOp�,sequential_3/dense_11/BiasAdd/ReadVariableOp�+sequential_3/dense_11/MatMul/ReadVariableOp�+sequential_3/dense_9/BiasAdd/ReadVariableOp�*sequential_3/dense_9/MatMul/ReadVariableOp[
sequential_3/reshape_3/ShapeShapereshape_3_input*
T0*
_output_shapes
:t
*sequential_3/reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_3/reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_3/reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$sequential_3/reshape_3/strided_sliceStridedSlice%sequential_3/reshape_3/Shape:output:03sequential_3/reshape_3/strided_slice/stack:output:05sequential_3/reshape_3/strided_slice/stack_1:output:05sequential_3/reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&sequential_3/reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :h
&sequential_3/reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :h
&sequential_3/reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :i
&sequential_3/reshape_3/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value
B :��
$sequential_3/reshape_3/Reshape/shapePack-sequential_3/reshape_3/strided_slice:output:0/sequential_3/reshape_3/Reshape/shape/1:output:0/sequential_3/reshape_3/Reshape/shape/2:output:0/sequential_3/reshape_3/Reshape/shape/3:output:0/sequential_3/reshape_3/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:�
sequential_3/reshape_3/ReshapeReshapereshape_3_input-sequential_3/reshape_3/Reshape/shape:output:0*
T0*4
_output_shapes"
 :�����������
%sequential_3/conv_lstm2d_3/zeros_like	ZerosLike'sequential_3/reshape_3/Reshape:output:0*
T0*4
_output_shapes"
 :����������r
0sequential_3/conv_lstm2d_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
sequential_3/conv_lstm2d_3/SumSum)sequential_3/conv_lstm2d_3/zeros_like:y:09sequential_3/conv_lstm2d_3/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:�����������
0sequential_3/conv_lstm2d_3/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"           k
&sequential_3/conv_lstm2d_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
 sequential_3/conv_lstm2d_3/zerosFill9sequential_3/conv_lstm2d_3/zeros/shape_as_tensor:output:0/sequential_3/conv_lstm2d_3/zeros/Const:output:0*
T0*'
_output_shapes
:��
&sequential_3/conv_lstm2d_3/convolutionConv2D'sequential_3/conv_lstm2d_3/Sum:output:0)sequential_3/conv_lstm2d_3/zeros:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
)sequential_3/conv_lstm2d_3/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                �
$sequential_3/conv_lstm2d_3/transpose	Transpose'sequential_3/reshape_3/Reshape:output:02sequential_3/conv_lstm2d_3/transpose/perm:output:0*
T0*4
_output_shapes"
 :����������x
 sequential_3/conv_lstm2d_3/ShapeShape(sequential_3/conv_lstm2d_3/transpose:y:0*
T0*
_output_shapes
:x
.sequential_3/conv_lstm2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0sequential_3/conv_lstm2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0sequential_3/conv_lstm2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(sequential_3/conv_lstm2d_3/strided_sliceStridedSlice)sequential_3/conv_lstm2d_3/Shape:output:07sequential_3/conv_lstm2d_3/strided_slice/stack:output:09sequential_3/conv_lstm2d_3/strided_slice/stack_1:output:09sequential_3/conv_lstm2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
6sequential_3/conv_lstm2d_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
(sequential_3/conv_lstm2d_3/TensorArrayV2TensorListReserve?sequential_3/conv_lstm2d_3/TensorArrayV2/element_shape:output:01sequential_3/conv_lstm2d_3/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Psequential_3/conv_lstm2d_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����        �
Bsequential_3/conv_lstm2d_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(sequential_3/conv_lstm2d_3/transpose:y:0Ysequential_3/conv_lstm2d_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���z
0sequential_3/conv_lstm2d_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_3/conv_lstm2d_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_3/conv_lstm2d_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*sequential_3/conv_lstm2d_3/strided_slice_1StridedSlice(sequential_3/conv_lstm2d_3/transpose:y:09sequential_3/conv_lstm2d_3/strided_slice_1/stack:output:0;sequential_3/conv_lstm2d_3/strided_slice_1/stack_1:output:0;sequential_3/conv_lstm2d_3/strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:����������*
shrink_axis_maskl
*sequential_3/conv_lstm2d_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
/sequential_3/conv_lstm2d_3/split/ReadVariableOpReadVariableOp8sequential_3_conv_lstm2d_3_split_readvariableop_resource*'
_output_shapes
:�4*
dtype0�
 sequential_3/conv_lstm2d_3/splitSplit3sequential_3/conv_lstm2d_3/split/split_dim:output:07sequential_3/conv_lstm2d_3/split/ReadVariableOp:value:0*
T0*`
_output_shapesN
L:�:�:�:�*
	num_splitn
,sequential_3/conv_lstm2d_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
1sequential_3/conv_lstm2d_3/split_1/ReadVariableOpReadVariableOp:sequential_3_conv_lstm2d_3_split_1_readvariableop_resource*&
_output_shapes
:4*
dtype0�
"sequential_3/conv_lstm2d_3/split_1Split5sequential_3/conv_lstm2d_3/split_1/split_dim:output:09sequential_3/conv_lstm2d_3/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitn
,sequential_3/conv_lstm2d_3/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
1sequential_3/conv_lstm2d_3/split_2/ReadVariableOpReadVariableOp:sequential_3_conv_lstm2d_3_split_2_readvariableop_resource*
_output_shapes
:4*
dtype0�
"sequential_3/conv_lstm2d_3/split_2Split5sequential_3/conv_lstm2d_3/split_2/split_dim:output:09sequential_3/conv_lstm2d_3/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split�
(sequential_3/conv_lstm2d_3/convolution_1Conv2D3sequential_3/conv_lstm2d_3/strided_slice_1:output:0)sequential_3/conv_lstm2d_3/split:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
"sequential_3/conv_lstm2d_3/BiasAddBiasAdd1sequential_3/conv_lstm2d_3/convolution_1:output:0+sequential_3/conv_lstm2d_3/split_2:output:0*
T0*/
_output_shapes
:����������
(sequential_3/conv_lstm2d_3/convolution_2Conv2D3sequential_3/conv_lstm2d_3/strided_slice_1:output:0)sequential_3/conv_lstm2d_3/split:output:1*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
$sequential_3/conv_lstm2d_3/BiasAdd_1BiasAdd1sequential_3/conv_lstm2d_3/convolution_2:output:0+sequential_3/conv_lstm2d_3/split_2:output:1*
T0*/
_output_shapes
:����������
(sequential_3/conv_lstm2d_3/convolution_3Conv2D3sequential_3/conv_lstm2d_3/strided_slice_1:output:0)sequential_3/conv_lstm2d_3/split:output:2*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
$sequential_3/conv_lstm2d_3/BiasAdd_2BiasAdd1sequential_3/conv_lstm2d_3/convolution_3:output:0+sequential_3/conv_lstm2d_3/split_2:output:2*
T0*/
_output_shapes
:����������
(sequential_3/conv_lstm2d_3/convolution_4Conv2D3sequential_3/conv_lstm2d_3/strided_slice_1:output:0)sequential_3/conv_lstm2d_3/split:output:3*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
$sequential_3/conv_lstm2d_3/BiasAdd_3BiasAdd1sequential_3/conv_lstm2d_3/convolution_4:output:0+sequential_3/conv_lstm2d_3/split_2:output:3*
T0*/
_output_shapes
:����������
(sequential_3/conv_lstm2d_3/convolution_5Conv2D/sequential_3/conv_lstm2d_3/convolution:output:0+sequential_3/conv_lstm2d_3/split_1:output:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
(sequential_3/conv_lstm2d_3/convolution_6Conv2D/sequential_3/conv_lstm2d_3/convolution:output:0+sequential_3/conv_lstm2d_3/split_1:output:1*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
(sequential_3/conv_lstm2d_3/convolution_7Conv2D/sequential_3/conv_lstm2d_3/convolution:output:0+sequential_3/conv_lstm2d_3/split_1:output:2*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
(sequential_3/conv_lstm2d_3/convolution_8Conv2D/sequential_3/conv_lstm2d_3/convolution:output:0+sequential_3/conv_lstm2d_3/split_1:output:3*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
sequential_3/conv_lstm2d_3/addAddV2+sequential_3/conv_lstm2d_3/BiasAdd:output:01sequential_3/conv_lstm2d_3/convolution_5:output:0*
T0*/
_output_shapes
:���������e
 sequential_3/conv_lstm2d_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>g
"sequential_3/conv_lstm2d_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?�
sequential_3/conv_lstm2d_3/MulMul"sequential_3/conv_lstm2d_3/add:z:0)sequential_3/conv_lstm2d_3/Const:output:0*
T0*/
_output_shapes
:����������
 sequential_3/conv_lstm2d_3/Add_1AddV2"sequential_3/conv_lstm2d_3/Mul:z:0+sequential_3/conv_lstm2d_3/Const_1:output:0*
T0*/
_output_shapes
:���������w
2sequential_3/conv_lstm2d_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
0sequential_3/conv_lstm2d_3/clip_by_value/MinimumMinimum$sequential_3/conv_lstm2d_3/Add_1:z:0;sequential_3/conv_lstm2d_3/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������o
*sequential_3/conv_lstm2d_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(sequential_3/conv_lstm2d_3/clip_by_valueMaximum4sequential_3/conv_lstm2d_3/clip_by_value/Minimum:z:03sequential_3/conv_lstm2d_3/clip_by_value/y:output:0*
T0*/
_output_shapes
:����������
 sequential_3/conv_lstm2d_3/add_2AddV2-sequential_3/conv_lstm2d_3/BiasAdd_1:output:01sequential_3/conv_lstm2d_3/convolution_6:output:0*
T0*/
_output_shapes
:���������g
"sequential_3/conv_lstm2d_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��L>g
"sequential_3/conv_lstm2d_3/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?�
 sequential_3/conv_lstm2d_3/Mul_1Mul$sequential_3/conv_lstm2d_3/add_2:z:0+sequential_3/conv_lstm2d_3/Const_2:output:0*
T0*/
_output_shapes
:����������
 sequential_3/conv_lstm2d_3/Add_3AddV2$sequential_3/conv_lstm2d_3/Mul_1:z:0+sequential_3/conv_lstm2d_3/Const_3:output:0*
T0*/
_output_shapes
:���������y
4sequential_3/conv_lstm2d_3/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
2sequential_3/conv_lstm2d_3/clip_by_value_1/MinimumMinimum$sequential_3/conv_lstm2d_3/Add_3:z:0=sequential_3/conv_lstm2d_3/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:���������q
,sequential_3/conv_lstm2d_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
*sequential_3/conv_lstm2d_3/clip_by_value_1Maximum6sequential_3/conv_lstm2d_3/clip_by_value_1/Minimum:z:05sequential_3/conv_lstm2d_3/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:����������
 sequential_3/conv_lstm2d_3/mul_2Mul.sequential_3/conv_lstm2d_3/clip_by_value_1:z:0/sequential_3/conv_lstm2d_3/convolution:output:0*
T0*/
_output_shapes
:����������
 sequential_3/conv_lstm2d_3/add_4AddV2-sequential_3/conv_lstm2d_3/BiasAdd_2:output:01sequential_3/conv_lstm2d_3/convolution_7:output:0*
T0*/
_output_shapes
:����������
sequential_3/conv_lstm2d_3/ReluRelu$sequential_3/conv_lstm2d_3/add_4:z:0*
T0*/
_output_shapes
:����������
 sequential_3/conv_lstm2d_3/mul_3Mul,sequential_3/conv_lstm2d_3/clip_by_value:z:0-sequential_3/conv_lstm2d_3/Relu:activations:0*
T0*/
_output_shapes
:����������
 sequential_3/conv_lstm2d_3/add_5AddV2$sequential_3/conv_lstm2d_3/mul_2:z:0$sequential_3/conv_lstm2d_3/mul_3:z:0*
T0*/
_output_shapes
:����������
 sequential_3/conv_lstm2d_3/add_6AddV2-sequential_3/conv_lstm2d_3/BiasAdd_3:output:01sequential_3/conv_lstm2d_3/convolution_8:output:0*
T0*/
_output_shapes
:���������g
"sequential_3/conv_lstm2d_3/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *��L>g
"sequential_3/conv_lstm2d_3/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?�
 sequential_3/conv_lstm2d_3/Mul_4Mul$sequential_3/conv_lstm2d_3/add_6:z:0+sequential_3/conv_lstm2d_3/Const_4:output:0*
T0*/
_output_shapes
:����������
 sequential_3/conv_lstm2d_3/Add_7AddV2$sequential_3/conv_lstm2d_3/Mul_4:z:0+sequential_3/conv_lstm2d_3/Const_5:output:0*
T0*/
_output_shapes
:���������y
4sequential_3/conv_lstm2d_3/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
2sequential_3/conv_lstm2d_3/clip_by_value_2/MinimumMinimum$sequential_3/conv_lstm2d_3/Add_7:z:0=sequential_3/conv_lstm2d_3/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:���������q
,sequential_3/conv_lstm2d_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
*sequential_3/conv_lstm2d_3/clip_by_value_2Maximum6sequential_3/conv_lstm2d_3/clip_by_value_2/Minimum:z:05sequential_3/conv_lstm2d_3/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:����������
!sequential_3/conv_lstm2d_3/Relu_1Relu$sequential_3/conv_lstm2d_3/add_5:z:0*
T0*/
_output_shapes
:����������
 sequential_3/conv_lstm2d_3/mul_5Mul.sequential_3/conv_lstm2d_3/clip_by_value_2:z:0/sequential_3/conv_lstm2d_3/Relu_1:activations:0*
T0*/
_output_shapes
:����������
8sequential_3/conv_lstm2d_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         y
7sequential_3/conv_lstm2d_3/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
*sequential_3/conv_lstm2d_3/TensorArrayV2_1TensorListReserveAsequential_3/conv_lstm2d_3/TensorArrayV2_1/element_shape:output:0@sequential_3/conv_lstm2d_3/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���a
sequential_3/conv_lstm2d_3/timeConst*
_output_shapes
: *
dtype0*
value	B : ~
3sequential_3/conv_lstm2d_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������o
-sequential_3/conv_lstm2d_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
 sequential_3/conv_lstm2d_3/whileWhile6sequential_3/conv_lstm2d_3/while/loop_counter:output:0<sequential_3/conv_lstm2d_3/while/maximum_iterations:output:0(sequential_3/conv_lstm2d_3/time:output:03sequential_3/conv_lstm2d_3/TensorArrayV2_1:handle:0/sequential_3/conv_lstm2d_3/convolution:output:0/sequential_3/conv_lstm2d_3/convolution:output:01sequential_3/conv_lstm2d_3/strided_slice:output:0Rsequential_3/conv_lstm2d_3/TensorArrayUnstack/TensorListFromTensor:output_handle:08sequential_3_conv_lstm2d_3_split_readvariableop_resource:sequential_3_conv_lstm2d_3_split_1_readvariableop_resource:sequential_3_conv_lstm2d_3_split_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *7
body/R-
+sequential_3_conv_lstm2d_3_while_body_68817*7
cond/R-
+sequential_3_conv_lstm2d_3_while_cond_68816*[
output_shapesJ
H: : : : :���������:���������: : : : : *
parallel_iterations �
Ksequential_3/conv_lstm2d_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
=sequential_3/conv_lstm2d_3/TensorArrayV2Stack/TensorListStackTensorListStack)sequential_3/conv_lstm2d_3/while:output:3Tsequential_3/conv_lstm2d_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:���������*
element_dtype0*
num_elements�
0sequential_3/conv_lstm2d_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������|
2sequential_3/conv_lstm2d_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: |
2sequential_3/conv_lstm2d_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*sequential_3/conv_lstm2d_3/strided_slice_2StridedSliceFsequential_3/conv_lstm2d_3/TensorArrayV2Stack/TensorListStack:tensor:09sequential_3/conv_lstm2d_3/strided_slice_2/stack:output:0;sequential_3/conv_lstm2d_3/strided_slice_2/stack_1:output:0;sequential_3/conv_lstm2d_3/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������*
shrink_axis_mask�
+sequential_3/conv_lstm2d_3/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                �
&sequential_3/conv_lstm2d_3/transpose_1	TransposeFsequential_3/conv_lstm2d_3/TensorArrayV2Stack/TensorListStack:tensor:04sequential_3/conv_lstm2d_3/transpose_1/perm:output:0*
T0*3
_output_shapes!
:����������
 sequential_3/dropout_12/IdentityIdentity3sequential_3/conv_lstm2d_3/strided_slice_2:output:0*
T0*/
_output_shapes
:����������
$sequential_3/max_pooling2d_3/MaxPoolMaxPool)sequential_3/dropout_12/Identity:output:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
�
 sequential_3/dropout_13/IdentityIdentity-sequential_3/max_pooling2d_3/MaxPool:output:0*
T0*/
_output_shapes
:���������m
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����'   �
sequential_3/flatten_3/ReshapeReshape)sequential_3/dropout_13/Identity:output:0%sequential_3/flatten_3/Const:output:0*
T0*'
_output_shapes
:���������'�
*sequential_3/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource*
_output_shapes
:	'�*
dtype0�
sequential_3/dense_9/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_3/dense_9/BiasAddBiasAdd%sequential_3/dense_9/MatMul:product:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
 sequential_3/dropout_14/IdentityIdentity'sequential_3/dense_9/Relu:activations:0*
T0*(
_output_shapes
:�����������
+sequential_3/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource*
_output_shapes
:	�2*
dtype0�
sequential_3/dense_10/MatMulMatMul)sequential_3/dropout_14/Identity:output:03sequential_3/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
sequential_3/dense_10/BiasAddBiasAdd&sequential_3/dense_10/MatMul:product:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2|
sequential_3/dense_10/ReluRelu&sequential_3/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:���������2�
 sequential_3/dropout_15/IdentityIdentity(sequential_3/dense_10/Relu:activations:0*
T0*'
_output_shapes
:���������2�
+sequential_3/dense_11/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_11_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
sequential_3/dense_11/MatMulMatMul)sequential_3/dropout_15/Identity:output:03sequential_3/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,sequential_3/dense_11/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_3/dense_11/BiasAddBiasAdd&sequential_3/dense_11/MatMul:product:04sequential_3/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
sequential_3/dense_11/SigmoidSigmoid&sequential_3/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:���������p
IdentityIdentity!sequential_3/dense_11/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp0^sequential_3/conv_lstm2d_3/split/ReadVariableOp2^sequential_3/conv_lstm2d_3/split_1/ReadVariableOp2^sequential_3/conv_lstm2d_3/split_2/ReadVariableOp!^sequential_3/conv_lstm2d_3/while-^sequential_3/dense_10/BiasAdd/ReadVariableOp,^sequential_3/dense_10/MatMul/ReadVariableOp-^sequential_3/dense_11/BiasAdd/ReadVariableOp,^sequential_3/dense_11/MatMul/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp+^sequential_3/dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 2b
/sequential_3/conv_lstm2d_3/split/ReadVariableOp/sequential_3/conv_lstm2d_3/split/ReadVariableOp2f
1sequential_3/conv_lstm2d_3/split_1/ReadVariableOp1sequential_3/conv_lstm2d_3/split_1/ReadVariableOp2f
1sequential_3/conv_lstm2d_3/split_2/ReadVariableOp1sequential_3/conv_lstm2d_3/split_2/ReadVariableOp2D
 sequential_3/conv_lstm2d_3/while sequential_3/conv_lstm2d_3/while2\
,sequential_3/dense_10/BiasAdd/ReadVariableOp,sequential_3/dense_10/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_10/MatMul/ReadVariableOp+sequential_3/dense_10/MatMul/ReadVariableOp2\
,sequential_3/dense_11/BiasAdd/ReadVariableOp,sequential_3/dense_11/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_11/MatMul/ReadVariableOp+sequential_3/dense_11/MatMul/ReadVariableOp2Z
+sequential_3/dense_9/BiasAdd/ReadVariableOp+sequential_3/dense_9/BiasAdd/ReadVariableOp2X
*sequential_3/dense_9/MatMul/ReadVariableOp*sequential_3/dense_9/MatMul/ReadVariableOp:] Y
,
_output_shapes
:����������
)
_user_specified_namereshape_3_input
�[
�
while_body_71115
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0@
%while_split_readvariableop_resource_0:�4A
'while_split_1_readvariableop_resource_0:45
'while_split_2_readvariableop_resource_0:4
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor>
#while_split_readvariableop_resource:�4?
%while_split_1_readvariableop_resource:43
%while_split_2_readvariableop_resource:4��while/split/ReadVariableOp�while/split_1/ReadVariableOp�while/split_2/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����        �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:����������*
element_dtype0W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:�4*
dtype0�
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*`
_output_shapesN
L:�:�:�:�*
	num_splitY
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
:4*
dtype0�
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes
:4*
dtype0�
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split�
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*/
_output_shapes
:����������
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*/
_output_shapes
:����������
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*/
_output_shapes
:����������
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*/
_output_shapes
:����������
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:���������P
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>R
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?o
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*/
_output_shapes
:���������u
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*/
_output_shapes
:���������b
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������Z
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:����������
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:���������R
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��L>R
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*/
_output_shapes
:���������w
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*/
_output_shapes
:���������d
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:���������\
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:���������|
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:����������
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:���������]

while/ReluReluwhile/add_4:z:0*
T0*/
_output_shapes
:���������
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*/
_output_shapes
:���������p
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:����������
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:���������R
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *��L>R
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*/
_output_shapes
:���������w
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*/
_output_shapes
:���������d
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:���������\
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:���������_
while/Relu_1Reluwhile/add_5:z:0*
T0*/
_output_shapes
:����������
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*/
_output_shapes
:���������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_5:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: O
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: [
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: t
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*/
_output_shapes
:���������t
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*/
_output_shapes
:����������

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :���������:���������: : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�3
�
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_69389

inputs"
unknown:�4#
	unknown_0:4
	unknown_1:4
identity��StatefulPartitionedCall�whileg

zeros_like	ZerosLikeinputs*
T0*=
_output_shapes+
):'�������������������W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :u
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*0
_output_shapes
:����������n
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"           P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    u
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
k
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'�������������������B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����        �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:����������*
shrink_axis_mask�
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0convolution:output:0convolution:output:0unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:���������:���������:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_conv_lstm_cell_3_layer_call_and_return_conditional_losses_69266v
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0unknown	unknown_0	unknown_1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_69320*
condR
while_cond_69319*[
output_shapesJ
H: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:���������*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������*
shrink_axis_maskm
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:���������o
IdentityIdentitystrided_slice_2:output:0^NoOp*
T0*/
_output_shapes
:���������h
NoOpNoOp^StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'�������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:e a
=
_output_shapes+
):'�������������������
 
_user_specified_nameinputs
�
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_69686

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����'   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������'X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������'"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
F
*__inference_dropout_12_layer_call_fn_71919

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_12_layer_call_and_return_conditional_losses_69670h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�e
�
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_71690

inputs8
split_readvariableop_resource:�49
split_1_readvariableop_resource:4-
split_2_readvariableop_resource:4
identity��split/ReadVariableOp�split_1/ReadVariableOp�split_2/ReadVariableOp�while^

zeros_like	ZerosLikeinputs*
T0*4
_output_shapes"
 :����������W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :u
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*0
_output_shapes
:����������n
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"           P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    u
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
k
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :����������B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����        �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:����������*
shrink_axis_maskQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :{
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:�4*
dtype0�
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*`
_output_shapesN
L:�:�:�:�*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:4*
dtype0�
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
:4*
dtype0�
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split�
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
v
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*/
_output_shapes
:����������
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*/
_output_shapes
:����������
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*/
_output_shapes
:����������
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*/
_output_shapes
:����������
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*/
_output_shapes
:���������J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:���������c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:���������\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:���������t
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*/
_output_shapes
:���������L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:���������e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:���������^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:���������V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:���������q
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:���������t
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*/
_output_shapes
:���������Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:���������m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:���������^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:���������t
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*/
_output_shapes
:���������L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:���������e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:���������^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:���������V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:���������S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:���������q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:���������v
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_71563*
condR
while_cond_71562*[
output_shapesJ
H: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:���������*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������*
shrink_axis_maskm
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:���������o
IdentityIdentitystrided_slice_2:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:����������: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�
�
conv_lstm2d_3_while_cond_707708
4conv_lstm2d_3_while_conv_lstm2d_3_while_loop_counter>
:conv_lstm2d_3_while_conv_lstm2d_3_while_maximum_iterations#
conv_lstm2d_3_while_placeholder%
!conv_lstm2d_3_while_placeholder_1%
!conv_lstm2d_3_while_placeholder_2%
!conv_lstm2d_3_while_placeholder_38
4conv_lstm2d_3_while_less_conv_lstm2d_3_strided_sliceO
Kconv_lstm2d_3_while_conv_lstm2d_3_while_cond_70770___redundant_placeholder0O
Kconv_lstm2d_3_while_conv_lstm2d_3_while_cond_70770___redundant_placeholder1O
Kconv_lstm2d_3_while_conv_lstm2d_3_while_cond_70770___redundant_placeholder2O
Kconv_lstm2d_3_while_conv_lstm2d_3_while_cond_70770___redundant_placeholder3 
conv_lstm2d_3_while_identity
�
conv_lstm2d_3/while/LessLessconv_lstm2d_3_while_placeholder4conv_lstm2d_3_while_less_conv_lstm2d_3_strided_slice*
T0*
_output_shapes
: g
conv_lstm2d_3/while/IdentityIdentityconv_lstm2d_3/while/Less:z:0*
T0
*
_output_shapes
: "E
conv_lstm2d_3_while_identity%conv_lstm2d_3/while/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :���������:���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�[
�
while_body_71563
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0@
%while_split_readvariableop_resource_0:�4A
'while_split_1_readvariableop_resource_0:45
'while_split_2_readvariableop_resource_0:4
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor>
#while_split_readvariableop_resource:�4?
%while_split_1_readvariableop_resource:43
%while_split_2_readvariableop_resource:4��while/split/ReadVariableOp�while/split_1/ReadVariableOp�while/split_2/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����        �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:����������*
element_dtype0W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:�4*
dtype0�
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*`
_output_shapesN
L:�:�:�:�*
	num_splitY
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
:4*
dtype0�
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes
:4*
dtype0�
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split�
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*/
_output_shapes
:����������
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*/
_output_shapes
:����������
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*/
_output_shapes
:����������
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*/
_output_shapes
:����������
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:���������P
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>R
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?o
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*/
_output_shapes
:���������u
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*/
_output_shapes
:���������b
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������Z
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:����������
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:���������R
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��L>R
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*/
_output_shapes
:���������w
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*/
_output_shapes
:���������d
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:���������\
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:���������|
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:����������
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:���������]

while/ReluReluwhile/add_4:z:0*
T0*/
_output_shapes
:���������
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*/
_output_shapes
:���������p
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:����������
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:���������R
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *��L>R
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*/
_output_shapes
:���������w
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*/
_output_shapes
:���������d
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:���������\
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:���������_
while/Relu_1Reluwhile/add_5:z:0*
T0*/
_output_shapes
:����������
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*/
_output_shapes
:���������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_5:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: O
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: [
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: t
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*/
_output_shapes
:���������t
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*/
_output_shapes
:����������

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :���������:���������: : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�

�
,__inference_sequential_3_layer_call_fn_70257
reshape_3_input"
unknown:�4#
	unknown_0:4
	unknown_1:4
	unknown_2:	'�
	unknown_3:	�
	unknown_4:	�2
	unknown_5:2
	unknown_6:2
	unknown_7:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallreshape_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_70213o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
,
_output_shapes
:����������
)
_user_specified_namereshape_3_input
�
c
E__inference_dropout_13_layer_call_and_return_conditional_losses_69678

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�e
�
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_71466
inputs_08
split_readvariableop_resource:�49
split_1_readvariableop_resource:4-
split_2_readvariableop_resource:4
identity��split/ReadVariableOp�split_1/ReadVariableOp�split_2/ReadVariableOp�whilei

zeros_like	ZerosLikeinputs_0*
T0*=
_output_shapes+
):'�������������������W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :u
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*0
_output_shapes
:����������n
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"           P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    u
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
k
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                �
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*=
_output_shapes+
):'�������������������B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����        �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:����������*
shrink_axis_maskQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :{
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:�4*
dtype0�
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*`
_output_shapesN
L:�:�:�:�*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:4*
dtype0�
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
:4*
dtype0�
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split�
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
v
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*/
_output_shapes
:����������
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*/
_output_shapes
:����������
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*/
_output_shapes
:����������
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*/
_output_shapes
:����������
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*/
_output_shapes
:���������J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:���������c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:���������\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:���������t
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*/
_output_shapes
:���������L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:���������e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:���������^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:���������V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:���������q
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:���������t
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*/
_output_shapes
:���������Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:���������m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:���������^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:���������t
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*/
_output_shapes
:���������L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:���������e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:���������^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:���������V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:���������S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:���������q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:���������v
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_71339*
condR
while_cond_71338*[
output_shapesJ
H: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:���������*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������*
shrink_axis_maskm
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:���������o
IdentityIdentitystrided_slice_2:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'�������������������: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:g c
=
_output_shapes+
):'�������������������
"
_user_specified_name
inputs/0
�
�
while_cond_69319
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_69319___redundant_placeholder03
/while_while_cond_69319___redundant_placeholder13
/while_while_cond_69319___redundant_placeholder23
/while_while_cond_69319___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :���������:���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�=
�
K__inference_conv_lstm_cell_3_layer_call_and_return_conditional_losses_69266

inputs

states
states_18
split_readvariableop_resource:�49
split_1_readvariableop_resource:4-
split_2_readvariableop_resource:4
identity

identity_1

identity_2��split/ReadVariableOp�split_1/ReadVariableOp�split_2/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :{
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:�4*
dtype0�
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*`
_output_shapesN
L:�:�:�:�*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:4*
dtype0�
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
:4*
dtype0�
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split�
convolutionConv2Dinputssplit:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
t
BiasAddBiasAddconvolution:output:0split_2:output:0*
T0*/
_output_shapes
:����������
convolution_1Conv2Dinputssplit:output:1*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_1BiasAddconvolution_1:output:0split_2:output:1*
T0*/
_output_shapes
:����������
convolution_2Conv2Dinputssplit:output:2*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_2BiasAddconvolution_2:output:0split_2:output:2*
T0*/
_output_shapes
:����������
convolution_3Conv2Dinputssplit:output:3*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_3BiasAddconvolution_3:output:0split_2:output:3*
T0*/
_output_shapes
:����������
convolution_4Conv2Dstatessplit_1:output:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_5Conv2Dstatessplit_1:output:1*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_6Conv2Dstatessplit_1:output:2*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_7Conv2Dstatessplit_1:output:3*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_4:output:0*
T0*/
_output_shapes
:���������J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:���������c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:���������\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:���������t
add_2AddV2BiasAdd_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:���������L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:���������e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:���������^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:���������V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:���������e
mul_2Mulclip_by_value_1:z:0states_1*
T0*/
_output_shapes
:���������t
add_4AddV2BiasAdd_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:���������Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:���������m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:���������^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:���������t
add_6AddV2BiasAdd_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:���������L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:���������e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:���������^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:���������V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:���������S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:���������q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:���������`
IdentityIdentity	mul_5:z:0^NoOp*
T0*/
_output_shapes
:���������b

Identity_1Identity	mul_5:z:0^NoOp*
T0*/
_output_shapes
:���������b

Identity_2Identity	add_5:z:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:���������:���������: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_namestates:WS
/
_output_shapes
:���������
 
_user_specified_namestates
�1
�
G__inference_sequential_3_layer_call_and_return_conditional_losses_70323
reshape_3_input.
conv_lstm2d_3_70294:�4-
conv_lstm2d_3_70296:4!
conv_lstm2d_3_70298:4 
dense_9_70305:	'�
dense_9_70307:	�!
dense_10_70311:	�2
dense_10_70313:2 
dense_11_70317:2
dense_11_70319:
identity��%conv_lstm2d_3/StatefulPartitionedCall� dense_10/StatefulPartitionedCall� dense_11/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�"dropout_12/StatefulPartitionedCall�"dropout_13/StatefulPartitionedCall�"dropout_14/StatefulPartitionedCall�"dropout_15/StatefulPartitionedCall�
reshape_3/PartitionedCallPartitionedCallreshape_3_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_69432�
%conv_lstm2d_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0conv_lstm2d_3_70294conv_lstm2d_3_70296conv_lstm2d_3_70298*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_70140�
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall.conv_lstm2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_12_layer_call_and_return_conditional_losses_69900�
max_pooling2d_3/PartitionedCallPartitionedCall+dropout_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_69407�
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_13_layer_call_and_return_conditional_losses_69877�
flatten_3/PartitionedCallPartitionedCall+dropout_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������'* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_69686�
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_9_70305dense_9_70307*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_69699�
"dropout_14/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0#^dropout_13/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_14_layer_call_and_return_conditional_losses_69838�
 dense_10/StatefulPartitionedCallStatefulPartitionedCall+dropout_14/StatefulPartitionedCall:output:0dense_10_70311dense_10_70313*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_69723�
"dropout_15/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0#^dropout_14/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_15_layer_call_and_return_conditional_losses_69805�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall+dropout_15/StatefulPartitionedCall:output:0dense_11_70317dense_11_70319*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_69747x
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp&^conv_lstm2d_3/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall#^dropout_14/StatefulPartitionedCall#^dropout_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 2N
%conv_lstm2d_3/StatefulPartitionedCall%conv_lstm2d_3/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2H
"dropout_14/StatefulPartitionedCall"dropout_14/StatefulPartitionedCall2H
"dropout_15/StatefulPartitionedCall"dropout_15/StatefulPartitionedCall:] Y
,
_output_shapes
:����������
)
_user_specified_namereshape_3_input
�1
�
G__inference_sequential_3_layer_call_and_return_conditional_losses_70213

inputs.
conv_lstm2d_3_70184:�4-
conv_lstm2d_3_70186:4!
conv_lstm2d_3_70188:4 
dense_9_70195:	'�
dense_9_70197:	�!
dense_10_70201:	�2
dense_10_70203:2 
dense_11_70207:2
dense_11_70209:
identity��%conv_lstm2d_3/StatefulPartitionedCall� dense_10/StatefulPartitionedCall� dense_11/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�"dropout_12/StatefulPartitionedCall�"dropout_13/StatefulPartitionedCall�"dropout_14/StatefulPartitionedCall�"dropout_15/StatefulPartitionedCall�
reshape_3/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_69432�
%conv_lstm2d_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0conv_lstm2d_3_70184conv_lstm2d_3_70186conv_lstm2d_3_70188*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_70140�
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall.conv_lstm2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_12_layer_call_and_return_conditional_losses_69900�
max_pooling2d_3/PartitionedCallPartitionedCall+dropout_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_69407�
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_13_layer_call_and_return_conditional_losses_69877�
flatten_3/PartitionedCallPartitionedCall+dropout_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������'* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_69686�
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_9_70195dense_9_70197*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_69699�
"dropout_14/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0#^dropout_13/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_14_layer_call_and_return_conditional_losses_69838�
 dense_10/StatefulPartitionedCallStatefulPartitionedCall+dropout_14/StatefulPartitionedCall:output:0dense_10_70201dense_10_70203*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_69723�
"dropout_15/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0#^dropout_14/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_15_layer_call_and_return_conditional_losses_69805�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall+dropout_15/StatefulPartitionedCall:output:0dense_11_70207dense_11_70209*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_69747x
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp&^conv_lstm2d_3/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall#^dropout_14/StatefulPartitionedCall#^dropout_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 2N
%conv_lstm2d_3/StatefulPartitionedCall%conv_lstm2d_3/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2H
"dropout_14/StatefulPartitionedCall"dropout_14/StatefulPartitionedCall2H
"dropout_15/StatefulPartitionedCall"dropout_15/StatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�[
�
while_body_69530
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0@
%while_split_readvariableop_resource_0:�4A
'while_split_1_readvariableop_resource_0:45
'while_split_2_readvariableop_resource_0:4
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor>
#while_split_readvariableop_resource:�4?
%while_split_1_readvariableop_resource:43
%while_split_2_readvariableop_resource:4��while/split/ReadVariableOp�while/split_1/ReadVariableOp�while/split_2/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����        �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:����������*
element_dtype0W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:�4*
dtype0�
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*`
_output_shapesN
L:�:�:�:�*
	num_splitY
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
:4*
dtype0�
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes
:4*
dtype0�
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split�
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*/
_output_shapes
:����������
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*/
_output_shapes
:����������
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*/
_output_shapes
:����������
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*/
_output_shapes
:����������
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:���������P
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>R
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?o
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*/
_output_shapes
:���������u
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*/
_output_shapes
:���������b
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������Z
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:����������
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:���������R
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��L>R
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*/
_output_shapes
:���������w
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*/
_output_shapes
:���������d
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:���������\
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:���������|
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:����������
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:���������]

while/ReluReluwhile/add_4:z:0*
T0*/
_output_shapes
:���������
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*/
_output_shapes
:���������p
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:����������
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:���������R
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *��L>R
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*/
_output_shapes
:���������w
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*/
_output_shapes
:���������d
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:���������\
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:���������_
while/Relu_1Reluwhile/add_5:z:0*
T0*/
_output_shapes
:����������
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*/
_output_shapes
:���������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_5:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: O
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: [
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: t
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*/
_output_shapes
:���������t
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*/
_output_shapes
:����������

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :���������:���������: : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�=
�
K__inference_conv_lstm_cell_3_layer_call_and_return_conditional_losses_72287

inputs
states_0
states_18
split_readvariableop_resource:�49
split_1_readvariableop_resource:4-
split_2_readvariableop_resource:4
identity

identity_1

identity_2��split/ReadVariableOp�split_1/ReadVariableOp�split_2/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :{
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:�4*
dtype0�
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*`
_output_shapesN
L:�:�:�:�*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:4*
dtype0�
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
:4*
dtype0�
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split�
convolutionConv2Dinputssplit:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
t
BiasAddBiasAddconvolution:output:0split_2:output:0*
T0*/
_output_shapes
:����������
convolution_1Conv2Dinputssplit:output:1*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_1BiasAddconvolution_1:output:0split_2:output:1*
T0*/
_output_shapes
:����������
convolution_2Conv2Dinputssplit:output:2*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_2BiasAddconvolution_2:output:0split_2:output:2*
T0*/
_output_shapes
:����������
convolution_3Conv2Dinputssplit:output:3*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_3BiasAddconvolution_3:output:0split_2:output:3*
T0*/
_output_shapes
:����������
convolution_4Conv2Dstates_0split_1:output:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_5Conv2Dstates_0split_1:output:1*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_6Conv2Dstates_0split_1:output:2*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_7Conv2Dstates_0split_1:output:3*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_4:output:0*
T0*/
_output_shapes
:���������J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:���������c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:���������\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:���������t
add_2AddV2BiasAdd_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:���������L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:���������e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:���������^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:���������V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:���������e
mul_2Mulclip_by_value_1:z:0states_1*
T0*/
_output_shapes
:���������t
add_4AddV2BiasAdd_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:���������Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:���������m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:���������^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:���������t
add_6AddV2BiasAdd_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:���������L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:���������e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:���������^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:���������V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:���������S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:���������q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:���������`
IdentityIdentity	mul_5:z:0^NoOp*
T0*/
_output_shapes
:���������b

Identity_1Identity	mul_5:z:0^NoOp*
T0*/
_output_shapes
:���������b

Identity_2Identity	add_5:z:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:���������:���������: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:YU
/
_output_shapes
:���������
"
_user_specified_name
states/0:YU
/
_output_shapes
:���������
"
_user_specified_name
states/1
�
c
E__inference_dropout_12_layer_call_and_return_conditional_losses_69670

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_71562
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_71562___redundant_placeholder03
/while_while_cond_71562___redundant_placeholder13
/while_while_cond_71562___redundant_placeholder23
/while_while_cond_71562___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :���������:���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
-__inference_conv_lstm2d_3_layer_call_fn_71007

inputs"
unknown:�4#
	unknown_0:4
	unknown_1:4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_69657w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:����������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�

d
E__inference_dropout_13_layer_call_and_return_conditional_losses_69877

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

d
E__inference_dropout_12_layer_call_and_return_conditional_losses_71941

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�u
�
conv_lstm2d_3_while_body_705088
4conv_lstm2d_3_while_conv_lstm2d_3_while_loop_counter>
:conv_lstm2d_3_while_conv_lstm2d_3_while_maximum_iterations#
conv_lstm2d_3_while_placeholder%
!conv_lstm2d_3_while_placeholder_1%
!conv_lstm2d_3_while_placeholder_2%
!conv_lstm2d_3_while_placeholder_35
1conv_lstm2d_3_while_conv_lstm2d_3_strided_slice_0s
oconv_lstm2d_3_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_3_tensorarrayunstack_tensorlistfromtensor_0N
3conv_lstm2d_3_while_split_readvariableop_resource_0:�4O
5conv_lstm2d_3_while_split_1_readvariableop_resource_0:4C
5conv_lstm2d_3_while_split_2_readvariableop_resource_0:4 
conv_lstm2d_3_while_identity"
conv_lstm2d_3_while_identity_1"
conv_lstm2d_3_while_identity_2"
conv_lstm2d_3_while_identity_3"
conv_lstm2d_3_while_identity_4"
conv_lstm2d_3_while_identity_53
/conv_lstm2d_3_while_conv_lstm2d_3_strided_sliceq
mconv_lstm2d_3_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_3_tensorarrayunstack_tensorlistfromtensorL
1conv_lstm2d_3_while_split_readvariableop_resource:�4M
3conv_lstm2d_3_while_split_1_readvariableop_resource:4A
3conv_lstm2d_3_while_split_2_readvariableop_resource:4��(conv_lstm2d_3/while/split/ReadVariableOp�*conv_lstm2d_3/while/split_1/ReadVariableOp�*conv_lstm2d_3/while/split_2/ReadVariableOp�
Econv_lstm2d_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����        �
7conv_lstm2d_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemoconv_lstm2d_3_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_3_tensorarrayunstack_tensorlistfromtensor_0conv_lstm2d_3_while_placeholderNconv_lstm2d_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:����������*
element_dtype0e
#conv_lstm2d_3/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
(conv_lstm2d_3/while/split/ReadVariableOpReadVariableOp3conv_lstm2d_3_while_split_readvariableop_resource_0*'
_output_shapes
:�4*
dtype0�
conv_lstm2d_3/while/splitSplit,conv_lstm2d_3/while/split/split_dim:output:00conv_lstm2d_3/while/split/ReadVariableOp:value:0*
T0*`
_output_shapesN
L:�:�:�:�*
	num_splitg
%conv_lstm2d_3/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
*conv_lstm2d_3/while/split_1/ReadVariableOpReadVariableOp5conv_lstm2d_3_while_split_1_readvariableop_resource_0*&
_output_shapes
:4*
dtype0�
conv_lstm2d_3/while/split_1Split.conv_lstm2d_3/while/split_1/split_dim:output:02conv_lstm2d_3/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitg
%conv_lstm2d_3/while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
*conv_lstm2d_3/while/split_2/ReadVariableOpReadVariableOp5conv_lstm2d_3_while_split_2_readvariableop_resource_0*
_output_shapes
:4*
dtype0�
conv_lstm2d_3/while/split_2Split.conv_lstm2d_3/while/split_2/split_dim:output:02conv_lstm2d_3/while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split�
conv_lstm2d_3/while/convolutionConv2D>conv_lstm2d_3/while/TensorArrayV2Read/TensorListGetItem:item:0"conv_lstm2d_3/while/split:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv_lstm2d_3/while/BiasAddBiasAdd(conv_lstm2d_3/while/convolution:output:0$conv_lstm2d_3/while/split_2:output:0*
T0*/
_output_shapes
:����������
!conv_lstm2d_3/while/convolution_1Conv2D>conv_lstm2d_3/while/TensorArrayV2Read/TensorListGetItem:item:0"conv_lstm2d_3/while/split:output:1*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv_lstm2d_3/while/BiasAdd_1BiasAdd*conv_lstm2d_3/while/convolution_1:output:0$conv_lstm2d_3/while/split_2:output:1*
T0*/
_output_shapes
:����������
!conv_lstm2d_3/while/convolution_2Conv2D>conv_lstm2d_3/while/TensorArrayV2Read/TensorListGetItem:item:0"conv_lstm2d_3/while/split:output:2*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv_lstm2d_3/while/BiasAdd_2BiasAdd*conv_lstm2d_3/while/convolution_2:output:0$conv_lstm2d_3/while/split_2:output:2*
T0*/
_output_shapes
:����������
!conv_lstm2d_3/while/convolution_3Conv2D>conv_lstm2d_3/while/TensorArrayV2Read/TensorListGetItem:item:0"conv_lstm2d_3/while/split:output:3*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv_lstm2d_3/while/BiasAdd_3BiasAdd*conv_lstm2d_3/while/convolution_3:output:0$conv_lstm2d_3/while/split_2:output:3*
T0*/
_output_shapes
:����������
!conv_lstm2d_3/while/convolution_4Conv2D!conv_lstm2d_3_while_placeholder_2$conv_lstm2d_3/while/split_1:output:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
!conv_lstm2d_3/while/convolution_5Conv2D!conv_lstm2d_3_while_placeholder_2$conv_lstm2d_3/while/split_1:output:1*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
!conv_lstm2d_3/while/convolution_6Conv2D!conv_lstm2d_3_while_placeholder_2$conv_lstm2d_3/while/split_1:output:2*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
!conv_lstm2d_3/while/convolution_7Conv2D!conv_lstm2d_3_while_placeholder_2$conv_lstm2d_3/while/split_1:output:3*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
conv_lstm2d_3/while/addAddV2$conv_lstm2d_3/while/BiasAdd:output:0*conv_lstm2d_3/while/convolution_4:output:0*
T0*/
_output_shapes
:���������^
conv_lstm2d_3/while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>`
conv_lstm2d_3/while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?�
conv_lstm2d_3/while/MulMulconv_lstm2d_3/while/add:z:0"conv_lstm2d_3/while/Const:output:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/while/Add_1AddV2conv_lstm2d_3/while/Mul:z:0$conv_lstm2d_3/while/Const_1:output:0*
T0*/
_output_shapes
:���������p
+conv_lstm2d_3/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
)conv_lstm2d_3/while/clip_by_value/MinimumMinimumconv_lstm2d_3/while/Add_1:z:04conv_lstm2d_3/while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������h
#conv_lstm2d_3/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
!conv_lstm2d_3/while/clip_by_valueMaximum-conv_lstm2d_3/while/clip_by_value/Minimum:z:0,conv_lstm2d_3/while/clip_by_value/y:output:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/while/add_2AddV2&conv_lstm2d_3/while/BiasAdd_1:output:0*conv_lstm2d_3/while/convolution_5:output:0*
T0*/
_output_shapes
:���������`
conv_lstm2d_3/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��L>`
conv_lstm2d_3/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?�
conv_lstm2d_3/while/Mul_1Mulconv_lstm2d_3/while/add_2:z:0$conv_lstm2d_3/while/Const_2:output:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/while/Add_3AddV2conv_lstm2d_3/while/Mul_1:z:0$conv_lstm2d_3/while/Const_3:output:0*
T0*/
_output_shapes
:���������r
-conv_lstm2d_3/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
+conv_lstm2d_3/while/clip_by_value_1/MinimumMinimumconv_lstm2d_3/while/Add_3:z:06conv_lstm2d_3/while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:���������j
%conv_lstm2d_3/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
#conv_lstm2d_3/while/clip_by_value_1Maximum/conv_lstm2d_3/while/clip_by_value_1/Minimum:z:0.conv_lstm2d_3/while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/while/mul_2Mul'conv_lstm2d_3/while/clip_by_value_1:z:0!conv_lstm2d_3_while_placeholder_3*
T0*/
_output_shapes
:����������
conv_lstm2d_3/while/add_4AddV2&conv_lstm2d_3/while/BiasAdd_2:output:0*conv_lstm2d_3/while/convolution_6:output:0*
T0*/
_output_shapes
:���������y
conv_lstm2d_3/while/ReluReluconv_lstm2d_3/while/add_4:z:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/while/mul_3Mul%conv_lstm2d_3/while/clip_by_value:z:0&conv_lstm2d_3/while/Relu:activations:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/while/add_5AddV2conv_lstm2d_3/while/mul_2:z:0conv_lstm2d_3/while/mul_3:z:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/while/add_6AddV2&conv_lstm2d_3/while/BiasAdd_3:output:0*conv_lstm2d_3/while/convolution_7:output:0*
T0*/
_output_shapes
:���������`
conv_lstm2d_3/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *��L>`
conv_lstm2d_3/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?�
conv_lstm2d_3/while/Mul_4Mulconv_lstm2d_3/while/add_6:z:0$conv_lstm2d_3/while/Const_4:output:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/while/Add_7AddV2conv_lstm2d_3/while/Mul_4:z:0$conv_lstm2d_3/while/Const_5:output:0*
T0*/
_output_shapes
:���������r
-conv_lstm2d_3/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
+conv_lstm2d_3/while/clip_by_value_2/MinimumMinimumconv_lstm2d_3/while/Add_7:z:06conv_lstm2d_3/while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:���������j
%conv_lstm2d_3/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
#conv_lstm2d_3/while/clip_by_value_2Maximum/conv_lstm2d_3/while/clip_by_value_2/Minimum:z:0.conv_lstm2d_3/while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:���������{
conv_lstm2d_3/while/Relu_1Reluconv_lstm2d_3/while/add_5:z:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/while/mul_5Mul'conv_lstm2d_3/while/clip_by_value_2:z:0(conv_lstm2d_3/while/Relu_1:activations:0*
T0*/
_output_shapes
:����������
>conv_lstm2d_3/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
8conv_lstm2d_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!conv_lstm2d_3_while_placeholder_1Gconv_lstm2d_3/while/TensorArrayV2Write/TensorListSetItem/index:output:0conv_lstm2d_3/while/mul_5:z:0*
_output_shapes
: *
element_dtype0:���]
conv_lstm2d_3/while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :�
conv_lstm2d_3/while/add_8AddV2conv_lstm2d_3_while_placeholder$conv_lstm2d_3/while/add_8/y:output:0*
T0*
_output_shapes
: ]
conv_lstm2d_3/while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :�
conv_lstm2d_3/while/add_9AddV24conv_lstm2d_3_while_conv_lstm2d_3_while_loop_counter$conv_lstm2d_3/while/add_9/y:output:0*
T0*
_output_shapes
: �
conv_lstm2d_3/while/IdentityIdentityconv_lstm2d_3/while/add_9:z:0^conv_lstm2d_3/while/NoOp*
T0*
_output_shapes
: �
conv_lstm2d_3/while/Identity_1Identity:conv_lstm2d_3_while_conv_lstm2d_3_while_maximum_iterations^conv_lstm2d_3/while/NoOp*
T0*
_output_shapes
: �
conv_lstm2d_3/while/Identity_2Identityconv_lstm2d_3/while/add_8:z:0^conv_lstm2d_3/while/NoOp*
T0*
_output_shapes
: �
conv_lstm2d_3/while/Identity_3IdentityHconv_lstm2d_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^conv_lstm2d_3/while/NoOp*
T0*
_output_shapes
: �
conv_lstm2d_3/while/Identity_4Identityconv_lstm2d_3/while/mul_5:z:0^conv_lstm2d_3/while/NoOp*
T0*/
_output_shapes
:����������
conv_lstm2d_3/while/Identity_5Identityconv_lstm2d_3/while/add_5:z:0^conv_lstm2d_3/while/NoOp*
T0*/
_output_shapes
:����������
conv_lstm2d_3/while/NoOpNoOp)^conv_lstm2d_3/while/split/ReadVariableOp+^conv_lstm2d_3/while/split_1/ReadVariableOp+^conv_lstm2d_3/while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/conv_lstm2d_3_while_conv_lstm2d_3_strided_slice1conv_lstm2d_3_while_conv_lstm2d_3_strided_slice_0"E
conv_lstm2d_3_while_identity%conv_lstm2d_3/while/Identity:output:0"I
conv_lstm2d_3_while_identity_1'conv_lstm2d_3/while/Identity_1:output:0"I
conv_lstm2d_3_while_identity_2'conv_lstm2d_3/while/Identity_2:output:0"I
conv_lstm2d_3_while_identity_3'conv_lstm2d_3/while/Identity_3:output:0"I
conv_lstm2d_3_while_identity_4'conv_lstm2d_3/while/Identity_4:output:0"I
conv_lstm2d_3_while_identity_5'conv_lstm2d_3/while/Identity_5:output:0"l
3conv_lstm2d_3_while_split_1_readvariableop_resource5conv_lstm2d_3_while_split_1_readvariableop_resource_0"l
3conv_lstm2d_3_while_split_2_readvariableop_resource5conv_lstm2d_3_while_split_2_readvariableop_resource_0"h
1conv_lstm2d_3_while_split_readvariableop_resource3conv_lstm2d_3_while_split_readvariableop_resource_0"�
mconv_lstm2d_3_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_3_tensorarrayunstack_tensorlistfromtensoroconv_lstm2d_3_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :���������:���������: : : : : 2T
(conv_lstm2d_3/while/split/ReadVariableOp(conv_lstm2d_3/while/split/ReadVariableOp2X
*conv_lstm2d_3/while/split_1/ReadVariableOp*conv_lstm2d_3/while/split_1/ReadVariableOp2X
*conv_lstm2d_3/while/split_2/ReadVariableOp*conv_lstm2d_3/while/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�	
d
E__inference_dropout_15_layer_call_and_return_conditional_losses_72083

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������2C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������2*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������2o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������2i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������2Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
c
E__inference_dropout_15_layer_call_and_return_conditional_losses_72071

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������2[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������2"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
E
)__inference_reshape_3_layer_call_fn_70959

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_69432m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

d
E__inference_dropout_12_layer_call_and_return_conditional_losses_69900

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
E__inference_dropout_12_layer_call_and_return_conditional_losses_71929

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
*__inference_dropout_12_layer_call_fn_71924

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_12_layer_call_and_return_conditional_losses_69900w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�e
�
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_71914

inputs8
split_readvariableop_resource:�49
split_1_readvariableop_resource:4-
split_2_readvariableop_resource:4
identity��split/ReadVariableOp�split_1/ReadVariableOp�split_2/ReadVariableOp�while^

zeros_like	ZerosLikeinputs*
T0*4
_output_shapes"
 :����������W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :u
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*0
_output_shapes
:����������n
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"           P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    u
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
k
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :����������B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����        �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:����������*
shrink_axis_maskQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :{
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:�4*
dtype0�
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*`
_output_shapesN
L:�:�:�:�*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:4*
dtype0�
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
:4*
dtype0�
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split�
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
v
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*/
_output_shapes
:����������
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*/
_output_shapes
:����������
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*/
_output_shapes
:����������
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*/
_output_shapes
:����������
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*/
_output_shapes
:���������J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:���������c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:���������\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:���������t
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*/
_output_shapes
:���������L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:���������e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:���������^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:���������V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:���������q
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:���������t
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*/
_output_shapes
:���������Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:���������m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:���������^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:���������t
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*/
_output_shapes
:���������L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:���������e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:���������^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:���������V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:���������S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:���������q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:���������v
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_71787*
condR
while_cond_71786*[
output_shapesJ
H: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:���������*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������*
shrink_axis_maskm
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:���������o
IdentityIdentitystrided_slice_2:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:����������: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�
�
while_cond_71338
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_71338___redundant_placeholder03
/while_while_cond_71338___redundant_placeholder13
/while_while_cond_71338___redundant_placeholder23
/while_while_cond_71338___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :���������:���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�e
�
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_71242
inputs_08
split_readvariableop_resource:�49
split_1_readvariableop_resource:4-
split_2_readvariableop_resource:4
identity��split/ReadVariableOp�split_1/ReadVariableOp�split_2/ReadVariableOp�whilei

zeros_like	ZerosLikeinputs_0*
T0*=
_output_shapes+
):'�������������������W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :u
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*0
_output_shapes
:����������n
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"           P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    u
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
k
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                �
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*=
_output_shapes+
):'�������������������B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����        �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:����������*
shrink_axis_maskQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :{
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:�4*
dtype0�
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*`
_output_shapesN
L:�:�:�:�*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:4*
dtype0�
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
:4*
dtype0�
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split�
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
v
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*/
_output_shapes
:����������
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*/
_output_shapes
:����������
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*/
_output_shapes
:����������
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*/
_output_shapes
:����������
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*/
_output_shapes
:���������J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:���������c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:���������\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:���������t
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*/
_output_shapes
:���������L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:���������e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:���������^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:���������V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:���������q
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:���������t
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*/
_output_shapes
:���������Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:���������m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:���������^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:���������t
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*/
_output_shapes
:���������L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:���������e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:���������^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:���������V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:���������S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:���������q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:���������v
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_71115*
condR
while_cond_71114*[
output_shapesJ
H: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:���������*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������*
shrink_axis_maskm
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:���������o
IdentityIdentitystrided_slice_2:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'�������������������: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:g c
=
_output_shapes+
):'�������������������
"
_user_specified_name
inputs/0
�
�
(__inference_dense_11_layer_call_fn_72092

inputs
unknown:2
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_69747o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�

�
C__inference_dense_10_layer_call_and_return_conditional_losses_72056

inputs1
matmul_readvariableop_resource:	�2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
while_cond_69529
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_69529___redundant_placeholder03
/while_while_cond_69529___redundant_placeholder13
/while_while_cond_69529___redundant_placeholder23
/while_while_cond_69529___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :���������:���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�"
�
while_body_69091
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
while_69115_0:�4'
while_69117_0:4
while_69119_0:4
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
while_69115:�4%
while_69117:4
while_69119:4��while/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����        �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:����������*
element_dtype0�
while/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_69115_0while_69117_0while_69119_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:���������:���������:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_conv_lstm_cell_3_layer_call_and_return_conditional_losses_69076r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0&while/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity&while/StatefulPartitionedCall:output:1^while/NoOp*
T0*/
_output_shapes
:����������
while/Identity_5Identity&while/StatefulPartitionedCall:output:2^while/NoOp*
T0*/
_output_shapes
:���������l

while/NoOpNoOp^while/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
while_69115while_69115_0"
while_69117while_69117_0"
while_69119while_69119_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :���������:���������: : : : : 2>
while/StatefulPartitionedCallwhile/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
`
D__inference_reshape_3_layer_call_and_return_conditional_losses_70974

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :R
Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:q
ReshapeReshapeinputsReshape/shape:output:0*
T0*4
_output_shapes"
 :����������e
IdentityIdentityReshape:output:0*
T0*4
_output_shapes"
 :����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
K
/__inference_max_pooling2d_3_layer_call_fn_71946

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_69407�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
͍
�
+sequential_3_conv_lstm2d_3_while_body_68817R
Nsequential_3_conv_lstm2d_3_while_sequential_3_conv_lstm2d_3_while_loop_counterX
Tsequential_3_conv_lstm2d_3_while_sequential_3_conv_lstm2d_3_while_maximum_iterations0
,sequential_3_conv_lstm2d_3_while_placeholder2
.sequential_3_conv_lstm2d_3_while_placeholder_12
.sequential_3_conv_lstm2d_3_while_placeholder_22
.sequential_3_conv_lstm2d_3_while_placeholder_3O
Ksequential_3_conv_lstm2d_3_while_sequential_3_conv_lstm2d_3_strided_slice_0�
�sequential_3_conv_lstm2d_3_while_tensorarrayv2read_tensorlistgetitem_sequential_3_conv_lstm2d_3_tensorarrayunstack_tensorlistfromtensor_0[
@sequential_3_conv_lstm2d_3_while_split_readvariableop_resource_0:�4\
Bsequential_3_conv_lstm2d_3_while_split_1_readvariableop_resource_0:4P
Bsequential_3_conv_lstm2d_3_while_split_2_readvariableop_resource_0:4-
)sequential_3_conv_lstm2d_3_while_identity/
+sequential_3_conv_lstm2d_3_while_identity_1/
+sequential_3_conv_lstm2d_3_while_identity_2/
+sequential_3_conv_lstm2d_3_while_identity_3/
+sequential_3_conv_lstm2d_3_while_identity_4/
+sequential_3_conv_lstm2d_3_while_identity_5M
Isequential_3_conv_lstm2d_3_while_sequential_3_conv_lstm2d_3_strided_slice�
�sequential_3_conv_lstm2d_3_while_tensorarrayv2read_tensorlistgetitem_sequential_3_conv_lstm2d_3_tensorarrayunstack_tensorlistfromtensorY
>sequential_3_conv_lstm2d_3_while_split_readvariableop_resource:�4Z
@sequential_3_conv_lstm2d_3_while_split_1_readvariableop_resource:4N
@sequential_3_conv_lstm2d_3_while_split_2_readvariableop_resource:4��5sequential_3/conv_lstm2d_3/while/split/ReadVariableOp�7sequential_3/conv_lstm2d_3/while/split_1/ReadVariableOp�7sequential_3/conv_lstm2d_3/while/split_2/ReadVariableOp�
Rsequential_3/conv_lstm2d_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����        �
Dsequential_3/conv_lstm2d_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_3_conv_lstm2d_3_while_tensorarrayv2read_tensorlistgetitem_sequential_3_conv_lstm2d_3_tensorarrayunstack_tensorlistfromtensor_0,sequential_3_conv_lstm2d_3_while_placeholder[sequential_3/conv_lstm2d_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:����������*
element_dtype0r
0sequential_3/conv_lstm2d_3/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
5sequential_3/conv_lstm2d_3/while/split/ReadVariableOpReadVariableOp@sequential_3_conv_lstm2d_3_while_split_readvariableop_resource_0*'
_output_shapes
:�4*
dtype0�
&sequential_3/conv_lstm2d_3/while/splitSplit9sequential_3/conv_lstm2d_3/while/split/split_dim:output:0=sequential_3/conv_lstm2d_3/while/split/ReadVariableOp:value:0*
T0*`
_output_shapesN
L:�:�:�:�*
	num_splitt
2sequential_3/conv_lstm2d_3/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
7sequential_3/conv_lstm2d_3/while/split_1/ReadVariableOpReadVariableOpBsequential_3_conv_lstm2d_3_while_split_1_readvariableop_resource_0*&
_output_shapes
:4*
dtype0�
(sequential_3/conv_lstm2d_3/while/split_1Split;sequential_3/conv_lstm2d_3/while/split_1/split_dim:output:0?sequential_3/conv_lstm2d_3/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitt
2sequential_3/conv_lstm2d_3/while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
7sequential_3/conv_lstm2d_3/while/split_2/ReadVariableOpReadVariableOpBsequential_3_conv_lstm2d_3_while_split_2_readvariableop_resource_0*
_output_shapes
:4*
dtype0�
(sequential_3/conv_lstm2d_3/while/split_2Split;sequential_3/conv_lstm2d_3/while/split_2/split_dim:output:0?sequential_3/conv_lstm2d_3/while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split�
,sequential_3/conv_lstm2d_3/while/convolutionConv2DKsequential_3/conv_lstm2d_3/while/TensorArrayV2Read/TensorListGetItem:item:0/sequential_3/conv_lstm2d_3/while/split:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
(sequential_3/conv_lstm2d_3/while/BiasAddBiasAdd5sequential_3/conv_lstm2d_3/while/convolution:output:01sequential_3/conv_lstm2d_3/while/split_2:output:0*
T0*/
_output_shapes
:����������
.sequential_3/conv_lstm2d_3/while/convolution_1Conv2DKsequential_3/conv_lstm2d_3/while/TensorArrayV2Read/TensorListGetItem:item:0/sequential_3/conv_lstm2d_3/while/split:output:1*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
*sequential_3/conv_lstm2d_3/while/BiasAdd_1BiasAdd7sequential_3/conv_lstm2d_3/while/convolution_1:output:01sequential_3/conv_lstm2d_3/while/split_2:output:1*
T0*/
_output_shapes
:����������
.sequential_3/conv_lstm2d_3/while/convolution_2Conv2DKsequential_3/conv_lstm2d_3/while/TensorArrayV2Read/TensorListGetItem:item:0/sequential_3/conv_lstm2d_3/while/split:output:2*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
*sequential_3/conv_lstm2d_3/while/BiasAdd_2BiasAdd7sequential_3/conv_lstm2d_3/while/convolution_2:output:01sequential_3/conv_lstm2d_3/while/split_2:output:2*
T0*/
_output_shapes
:����������
.sequential_3/conv_lstm2d_3/while/convolution_3Conv2DKsequential_3/conv_lstm2d_3/while/TensorArrayV2Read/TensorListGetItem:item:0/sequential_3/conv_lstm2d_3/while/split:output:3*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
*sequential_3/conv_lstm2d_3/while/BiasAdd_3BiasAdd7sequential_3/conv_lstm2d_3/while/convolution_3:output:01sequential_3/conv_lstm2d_3/while/split_2:output:3*
T0*/
_output_shapes
:����������
.sequential_3/conv_lstm2d_3/while/convolution_4Conv2D.sequential_3_conv_lstm2d_3_while_placeholder_21sequential_3/conv_lstm2d_3/while/split_1:output:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
.sequential_3/conv_lstm2d_3/while/convolution_5Conv2D.sequential_3_conv_lstm2d_3_while_placeholder_21sequential_3/conv_lstm2d_3/while/split_1:output:1*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
.sequential_3/conv_lstm2d_3/while/convolution_6Conv2D.sequential_3_conv_lstm2d_3_while_placeholder_21sequential_3/conv_lstm2d_3/while/split_1:output:2*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
.sequential_3/conv_lstm2d_3/while/convolution_7Conv2D.sequential_3_conv_lstm2d_3_while_placeholder_21sequential_3/conv_lstm2d_3/while/split_1:output:3*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
$sequential_3/conv_lstm2d_3/while/addAddV21sequential_3/conv_lstm2d_3/while/BiasAdd:output:07sequential_3/conv_lstm2d_3/while/convolution_4:output:0*
T0*/
_output_shapes
:���������k
&sequential_3/conv_lstm2d_3/while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>m
(sequential_3/conv_lstm2d_3/while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?�
$sequential_3/conv_lstm2d_3/while/MulMul(sequential_3/conv_lstm2d_3/while/add:z:0/sequential_3/conv_lstm2d_3/while/Const:output:0*
T0*/
_output_shapes
:����������
&sequential_3/conv_lstm2d_3/while/Add_1AddV2(sequential_3/conv_lstm2d_3/while/Mul:z:01sequential_3/conv_lstm2d_3/while/Const_1:output:0*
T0*/
_output_shapes
:���������}
8sequential_3/conv_lstm2d_3/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
6sequential_3/conv_lstm2d_3/while/clip_by_value/MinimumMinimum*sequential_3/conv_lstm2d_3/while/Add_1:z:0Asequential_3/conv_lstm2d_3/while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������u
0sequential_3/conv_lstm2d_3/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.sequential_3/conv_lstm2d_3/while/clip_by_valueMaximum:sequential_3/conv_lstm2d_3/while/clip_by_value/Minimum:z:09sequential_3/conv_lstm2d_3/while/clip_by_value/y:output:0*
T0*/
_output_shapes
:����������
&sequential_3/conv_lstm2d_3/while/add_2AddV23sequential_3/conv_lstm2d_3/while/BiasAdd_1:output:07sequential_3/conv_lstm2d_3/while/convolution_5:output:0*
T0*/
_output_shapes
:���������m
(sequential_3/conv_lstm2d_3/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��L>m
(sequential_3/conv_lstm2d_3/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?�
&sequential_3/conv_lstm2d_3/while/Mul_1Mul*sequential_3/conv_lstm2d_3/while/add_2:z:01sequential_3/conv_lstm2d_3/while/Const_2:output:0*
T0*/
_output_shapes
:����������
&sequential_3/conv_lstm2d_3/while/Add_3AddV2*sequential_3/conv_lstm2d_3/while/Mul_1:z:01sequential_3/conv_lstm2d_3/while/Const_3:output:0*
T0*/
_output_shapes
:���������
:sequential_3/conv_lstm2d_3/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
8sequential_3/conv_lstm2d_3/while/clip_by_value_1/MinimumMinimum*sequential_3/conv_lstm2d_3/while/Add_3:z:0Csequential_3/conv_lstm2d_3/while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:���������w
2sequential_3/conv_lstm2d_3/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
0sequential_3/conv_lstm2d_3/while/clip_by_value_1Maximum<sequential_3/conv_lstm2d_3/while/clip_by_value_1/Minimum:z:0;sequential_3/conv_lstm2d_3/while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:����������
&sequential_3/conv_lstm2d_3/while/mul_2Mul4sequential_3/conv_lstm2d_3/while/clip_by_value_1:z:0.sequential_3_conv_lstm2d_3_while_placeholder_3*
T0*/
_output_shapes
:����������
&sequential_3/conv_lstm2d_3/while/add_4AddV23sequential_3/conv_lstm2d_3/while/BiasAdd_2:output:07sequential_3/conv_lstm2d_3/while/convolution_6:output:0*
T0*/
_output_shapes
:����������
%sequential_3/conv_lstm2d_3/while/ReluRelu*sequential_3/conv_lstm2d_3/while/add_4:z:0*
T0*/
_output_shapes
:����������
&sequential_3/conv_lstm2d_3/while/mul_3Mul2sequential_3/conv_lstm2d_3/while/clip_by_value:z:03sequential_3/conv_lstm2d_3/while/Relu:activations:0*
T0*/
_output_shapes
:����������
&sequential_3/conv_lstm2d_3/while/add_5AddV2*sequential_3/conv_lstm2d_3/while/mul_2:z:0*sequential_3/conv_lstm2d_3/while/mul_3:z:0*
T0*/
_output_shapes
:����������
&sequential_3/conv_lstm2d_3/while/add_6AddV23sequential_3/conv_lstm2d_3/while/BiasAdd_3:output:07sequential_3/conv_lstm2d_3/while/convolution_7:output:0*
T0*/
_output_shapes
:���������m
(sequential_3/conv_lstm2d_3/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *��L>m
(sequential_3/conv_lstm2d_3/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?�
&sequential_3/conv_lstm2d_3/while/Mul_4Mul*sequential_3/conv_lstm2d_3/while/add_6:z:01sequential_3/conv_lstm2d_3/while/Const_4:output:0*
T0*/
_output_shapes
:����������
&sequential_3/conv_lstm2d_3/while/Add_7AddV2*sequential_3/conv_lstm2d_3/while/Mul_4:z:01sequential_3/conv_lstm2d_3/while/Const_5:output:0*
T0*/
_output_shapes
:���������
:sequential_3/conv_lstm2d_3/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
8sequential_3/conv_lstm2d_3/while/clip_by_value_2/MinimumMinimum*sequential_3/conv_lstm2d_3/while/Add_7:z:0Csequential_3/conv_lstm2d_3/while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:���������w
2sequential_3/conv_lstm2d_3/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
0sequential_3/conv_lstm2d_3/while/clip_by_value_2Maximum<sequential_3/conv_lstm2d_3/while/clip_by_value_2/Minimum:z:0;sequential_3/conv_lstm2d_3/while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:����������
'sequential_3/conv_lstm2d_3/while/Relu_1Relu*sequential_3/conv_lstm2d_3/while/add_5:z:0*
T0*/
_output_shapes
:����������
&sequential_3/conv_lstm2d_3/while/mul_5Mul4sequential_3/conv_lstm2d_3/while/clip_by_value_2:z:05sequential_3/conv_lstm2d_3/while/Relu_1:activations:0*
T0*/
_output_shapes
:����������
Ksequential_3/conv_lstm2d_3/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
Esequential_3/conv_lstm2d_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem.sequential_3_conv_lstm2d_3_while_placeholder_1Tsequential_3/conv_lstm2d_3/while/TensorArrayV2Write/TensorListSetItem/index:output:0*sequential_3/conv_lstm2d_3/while/mul_5:z:0*
_output_shapes
: *
element_dtype0:���j
(sequential_3/conv_lstm2d_3/while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :�
&sequential_3/conv_lstm2d_3/while/add_8AddV2,sequential_3_conv_lstm2d_3_while_placeholder1sequential_3/conv_lstm2d_3/while/add_8/y:output:0*
T0*
_output_shapes
: j
(sequential_3/conv_lstm2d_3/while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :�
&sequential_3/conv_lstm2d_3/while/add_9AddV2Nsequential_3_conv_lstm2d_3_while_sequential_3_conv_lstm2d_3_while_loop_counter1sequential_3/conv_lstm2d_3/while/add_9/y:output:0*
T0*
_output_shapes
: �
)sequential_3/conv_lstm2d_3/while/IdentityIdentity*sequential_3/conv_lstm2d_3/while/add_9:z:0&^sequential_3/conv_lstm2d_3/while/NoOp*
T0*
_output_shapes
: �
+sequential_3/conv_lstm2d_3/while/Identity_1IdentityTsequential_3_conv_lstm2d_3_while_sequential_3_conv_lstm2d_3_while_maximum_iterations&^sequential_3/conv_lstm2d_3/while/NoOp*
T0*
_output_shapes
: �
+sequential_3/conv_lstm2d_3/while/Identity_2Identity*sequential_3/conv_lstm2d_3/while/add_8:z:0&^sequential_3/conv_lstm2d_3/while/NoOp*
T0*
_output_shapes
: �
+sequential_3/conv_lstm2d_3/while/Identity_3IdentityUsequential_3/conv_lstm2d_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0&^sequential_3/conv_lstm2d_3/while/NoOp*
T0*
_output_shapes
: �
+sequential_3/conv_lstm2d_3/while/Identity_4Identity*sequential_3/conv_lstm2d_3/while/mul_5:z:0&^sequential_3/conv_lstm2d_3/while/NoOp*
T0*/
_output_shapes
:����������
+sequential_3/conv_lstm2d_3/while/Identity_5Identity*sequential_3/conv_lstm2d_3/while/add_5:z:0&^sequential_3/conv_lstm2d_3/while/NoOp*
T0*/
_output_shapes
:����������
%sequential_3/conv_lstm2d_3/while/NoOpNoOp6^sequential_3/conv_lstm2d_3/while/split/ReadVariableOp8^sequential_3/conv_lstm2d_3/while/split_1/ReadVariableOp8^sequential_3/conv_lstm2d_3/while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "_
)sequential_3_conv_lstm2d_3_while_identity2sequential_3/conv_lstm2d_3/while/Identity:output:0"c
+sequential_3_conv_lstm2d_3_while_identity_14sequential_3/conv_lstm2d_3/while/Identity_1:output:0"c
+sequential_3_conv_lstm2d_3_while_identity_24sequential_3/conv_lstm2d_3/while/Identity_2:output:0"c
+sequential_3_conv_lstm2d_3_while_identity_34sequential_3/conv_lstm2d_3/while/Identity_3:output:0"c
+sequential_3_conv_lstm2d_3_while_identity_44sequential_3/conv_lstm2d_3/while/Identity_4:output:0"c
+sequential_3_conv_lstm2d_3_while_identity_54sequential_3/conv_lstm2d_3/while/Identity_5:output:0"�
Isequential_3_conv_lstm2d_3_while_sequential_3_conv_lstm2d_3_strided_sliceKsequential_3_conv_lstm2d_3_while_sequential_3_conv_lstm2d_3_strided_slice_0"�
@sequential_3_conv_lstm2d_3_while_split_1_readvariableop_resourceBsequential_3_conv_lstm2d_3_while_split_1_readvariableop_resource_0"�
@sequential_3_conv_lstm2d_3_while_split_2_readvariableop_resourceBsequential_3_conv_lstm2d_3_while_split_2_readvariableop_resource_0"�
>sequential_3_conv_lstm2d_3_while_split_readvariableop_resource@sequential_3_conv_lstm2d_3_while_split_readvariableop_resource_0"�
�sequential_3_conv_lstm2d_3_while_tensorarrayv2read_tensorlistgetitem_sequential_3_conv_lstm2d_3_tensorarrayunstack_tensorlistfromtensor�sequential_3_conv_lstm2d_3_while_tensorarrayv2read_tensorlistgetitem_sequential_3_conv_lstm2d_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :���������:���������: : : : : 2n
5sequential_3/conv_lstm2d_3/while/split/ReadVariableOp5sequential_3/conv_lstm2d_3/while/split/ReadVariableOp2r
7sequential_3/conv_lstm2d_3/while/split_1/ReadVariableOp7sequential_3/conv_lstm2d_3/while/split_1/ReadVariableOp2r
7sequential_3/conv_lstm2d_3/while/split_2/ReadVariableOp7sequential_3/conv_lstm2d_3/while/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�u
�
conv_lstm2d_3_while_body_707718
4conv_lstm2d_3_while_conv_lstm2d_3_while_loop_counter>
:conv_lstm2d_3_while_conv_lstm2d_3_while_maximum_iterations#
conv_lstm2d_3_while_placeholder%
!conv_lstm2d_3_while_placeholder_1%
!conv_lstm2d_3_while_placeholder_2%
!conv_lstm2d_3_while_placeholder_35
1conv_lstm2d_3_while_conv_lstm2d_3_strided_slice_0s
oconv_lstm2d_3_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_3_tensorarrayunstack_tensorlistfromtensor_0N
3conv_lstm2d_3_while_split_readvariableop_resource_0:�4O
5conv_lstm2d_3_while_split_1_readvariableop_resource_0:4C
5conv_lstm2d_3_while_split_2_readvariableop_resource_0:4 
conv_lstm2d_3_while_identity"
conv_lstm2d_3_while_identity_1"
conv_lstm2d_3_while_identity_2"
conv_lstm2d_3_while_identity_3"
conv_lstm2d_3_while_identity_4"
conv_lstm2d_3_while_identity_53
/conv_lstm2d_3_while_conv_lstm2d_3_strided_sliceq
mconv_lstm2d_3_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_3_tensorarrayunstack_tensorlistfromtensorL
1conv_lstm2d_3_while_split_readvariableop_resource:�4M
3conv_lstm2d_3_while_split_1_readvariableop_resource:4A
3conv_lstm2d_3_while_split_2_readvariableop_resource:4��(conv_lstm2d_3/while/split/ReadVariableOp�*conv_lstm2d_3/while/split_1/ReadVariableOp�*conv_lstm2d_3/while/split_2/ReadVariableOp�
Econv_lstm2d_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����        �
7conv_lstm2d_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemoconv_lstm2d_3_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_3_tensorarrayunstack_tensorlistfromtensor_0conv_lstm2d_3_while_placeholderNconv_lstm2d_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:����������*
element_dtype0e
#conv_lstm2d_3/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
(conv_lstm2d_3/while/split/ReadVariableOpReadVariableOp3conv_lstm2d_3_while_split_readvariableop_resource_0*'
_output_shapes
:�4*
dtype0�
conv_lstm2d_3/while/splitSplit,conv_lstm2d_3/while/split/split_dim:output:00conv_lstm2d_3/while/split/ReadVariableOp:value:0*
T0*`
_output_shapesN
L:�:�:�:�*
	num_splitg
%conv_lstm2d_3/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
*conv_lstm2d_3/while/split_1/ReadVariableOpReadVariableOp5conv_lstm2d_3_while_split_1_readvariableop_resource_0*&
_output_shapes
:4*
dtype0�
conv_lstm2d_3/while/split_1Split.conv_lstm2d_3/while/split_1/split_dim:output:02conv_lstm2d_3/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitg
%conv_lstm2d_3/while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
*conv_lstm2d_3/while/split_2/ReadVariableOpReadVariableOp5conv_lstm2d_3_while_split_2_readvariableop_resource_0*
_output_shapes
:4*
dtype0�
conv_lstm2d_3/while/split_2Split.conv_lstm2d_3/while/split_2/split_dim:output:02conv_lstm2d_3/while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split�
conv_lstm2d_3/while/convolutionConv2D>conv_lstm2d_3/while/TensorArrayV2Read/TensorListGetItem:item:0"conv_lstm2d_3/while/split:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv_lstm2d_3/while/BiasAddBiasAdd(conv_lstm2d_3/while/convolution:output:0$conv_lstm2d_3/while/split_2:output:0*
T0*/
_output_shapes
:����������
!conv_lstm2d_3/while/convolution_1Conv2D>conv_lstm2d_3/while/TensorArrayV2Read/TensorListGetItem:item:0"conv_lstm2d_3/while/split:output:1*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv_lstm2d_3/while/BiasAdd_1BiasAdd*conv_lstm2d_3/while/convolution_1:output:0$conv_lstm2d_3/while/split_2:output:1*
T0*/
_output_shapes
:����������
!conv_lstm2d_3/while/convolution_2Conv2D>conv_lstm2d_3/while/TensorArrayV2Read/TensorListGetItem:item:0"conv_lstm2d_3/while/split:output:2*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv_lstm2d_3/while/BiasAdd_2BiasAdd*conv_lstm2d_3/while/convolution_2:output:0$conv_lstm2d_3/while/split_2:output:2*
T0*/
_output_shapes
:����������
!conv_lstm2d_3/while/convolution_3Conv2D>conv_lstm2d_3/while/TensorArrayV2Read/TensorListGetItem:item:0"conv_lstm2d_3/while/split:output:3*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv_lstm2d_3/while/BiasAdd_3BiasAdd*conv_lstm2d_3/while/convolution_3:output:0$conv_lstm2d_3/while/split_2:output:3*
T0*/
_output_shapes
:����������
!conv_lstm2d_3/while/convolution_4Conv2D!conv_lstm2d_3_while_placeholder_2$conv_lstm2d_3/while/split_1:output:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
!conv_lstm2d_3/while/convolution_5Conv2D!conv_lstm2d_3_while_placeholder_2$conv_lstm2d_3/while/split_1:output:1*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
!conv_lstm2d_3/while/convolution_6Conv2D!conv_lstm2d_3_while_placeholder_2$conv_lstm2d_3/while/split_1:output:2*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
!conv_lstm2d_3/while/convolution_7Conv2D!conv_lstm2d_3_while_placeholder_2$conv_lstm2d_3/while/split_1:output:3*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
conv_lstm2d_3/while/addAddV2$conv_lstm2d_3/while/BiasAdd:output:0*conv_lstm2d_3/while/convolution_4:output:0*
T0*/
_output_shapes
:���������^
conv_lstm2d_3/while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>`
conv_lstm2d_3/while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?�
conv_lstm2d_3/while/MulMulconv_lstm2d_3/while/add:z:0"conv_lstm2d_3/while/Const:output:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/while/Add_1AddV2conv_lstm2d_3/while/Mul:z:0$conv_lstm2d_3/while/Const_1:output:0*
T0*/
_output_shapes
:���������p
+conv_lstm2d_3/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
)conv_lstm2d_3/while/clip_by_value/MinimumMinimumconv_lstm2d_3/while/Add_1:z:04conv_lstm2d_3/while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������h
#conv_lstm2d_3/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
!conv_lstm2d_3/while/clip_by_valueMaximum-conv_lstm2d_3/while/clip_by_value/Minimum:z:0,conv_lstm2d_3/while/clip_by_value/y:output:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/while/add_2AddV2&conv_lstm2d_3/while/BiasAdd_1:output:0*conv_lstm2d_3/while/convolution_5:output:0*
T0*/
_output_shapes
:���������`
conv_lstm2d_3/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��L>`
conv_lstm2d_3/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?�
conv_lstm2d_3/while/Mul_1Mulconv_lstm2d_3/while/add_2:z:0$conv_lstm2d_3/while/Const_2:output:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/while/Add_3AddV2conv_lstm2d_3/while/Mul_1:z:0$conv_lstm2d_3/while/Const_3:output:0*
T0*/
_output_shapes
:���������r
-conv_lstm2d_3/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
+conv_lstm2d_3/while/clip_by_value_1/MinimumMinimumconv_lstm2d_3/while/Add_3:z:06conv_lstm2d_3/while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:���������j
%conv_lstm2d_3/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
#conv_lstm2d_3/while/clip_by_value_1Maximum/conv_lstm2d_3/while/clip_by_value_1/Minimum:z:0.conv_lstm2d_3/while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/while/mul_2Mul'conv_lstm2d_3/while/clip_by_value_1:z:0!conv_lstm2d_3_while_placeholder_3*
T0*/
_output_shapes
:����������
conv_lstm2d_3/while/add_4AddV2&conv_lstm2d_3/while/BiasAdd_2:output:0*conv_lstm2d_3/while/convolution_6:output:0*
T0*/
_output_shapes
:���������y
conv_lstm2d_3/while/ReluReluconv_lstm2d_3/while/add_4:z:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/while/mul_3Mul%conv_lstm2d_3/while/clip_by_value:z:0&conv_lstm2d_3/while/Relu:activations:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/while/add_5AddV2conv_lstm2d_3/while/mul_2:z:0conv_lstm2d_3/while/mul_3:z:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/while/add_6AddV2&conv_lstm2d_3/while/BiasAdd_3:output:0*conv_lstm2d_3/while/convolution_7:output:0*
T0*/
_output_shapes
:���������`
conv_lstm2d_3/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *��L>`
conv_lstm2d_3/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?�
conv_lstm2d_3/while/Mul_4Mulconv_lstm2d_3/while/add_6:z:0$conv_lstm2d_3/while/Const_4:output:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/while/Add_7AddV2conv_lstm2d_3/while/Mul_4:z:0$conv_lstm2d_3/while/Const_5:output:0*
T0*/
_output_shapes
:���������r
-conv_lstm2d_3/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
+conv_lstm2d_3/while/clip_by_value_2/MinimumMinimumconv_lstm2d_3/while/Add_7:z:06conv_lstm2d_3/while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:���������j
%conv_lstm2d_3/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
#conv_lstm2d_3/while/clip_by_value_2Maximum/conv_lstm2d_3/while/clip_by_value_2/Minimum:z:0.conv_lstm2d_3/while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:���������{
conv_lstm2d_3/while/Relu_1Reluconv_lstm2d_3/while/add_5:z:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/while/mul_5Mul'conv_lstm2d_3/while/clip_by_value_2:z:0(conv_lstm2d_3/while/Relu_1:activations:0*
T0*/
_output_shapes
:����������
>conv_lstm2d_3/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
8conv_lstm2d_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!conv_lstm2d_3_while_placeholder_1Gconv_lstm2d_3/while/TensorArrayV2Write/TensorListSetItem/index:output:0conv_lstm2d_3/while/mul_5:z:0*
_output_shapes
: *
element_dtype0:���]
conv_lstm2d_3/while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :�
conv_lstm2d_3/while/add_8AddV2conv_lstm2d_3_while_placeholder$conv_lstm2d_3/while/add_8/y:output:0*
T0*
_output_shapes
: ]
conv_lstm2d_3/while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :�
conv_lstm2d_3/while/add_9AddV24conv_lstm2d_3_while_conv_lstm2d_3_while_loop_counter$conv_lstm2d_3/while/add_9/y:output:0*
T0*
_output_shapes
: �
conv_lstm2d_3/while/IdentityIdentityconv_lstm2d_3/while/add_9:z:0^conv_lstm2d_3/while/NoOp*
T0*
_output_shapes
: �
conv_lstm2d_3/while/Identity_1Identity:conv_lstm2d_3_while_conv_lstm2d_3_while_maximum_iterations^conv_lstm2d_3/while/NoOp*
T0*
_output_shapes
: �
conv_lstm2d_3/while/Identity_2Identityconv_lstm2d_3/while/add_8:z:0^conv_lstm2d_3/while/NoOp*
T0*
_output_shapes
: �
conv_lstm2d_3/while/Identity_3IdentityHconv_lstm2d_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^conv_lstm2d_3/while/NoOp*
T0*
_output_shapes
: �
conv_lstm2d_3/while/Identity_4Identityconv_lstm2d_3/while/mul_5:z:0^conv_lstm2d_3/while/NoOp*
T0*/
_output_shapes
:����������
conv_lstm2d_3/while/Identity_5Identityconv_lstm2d_3/while/add_5:z:0^conv_lstm2d_3/while/NoOp*
T0*/
_output_shapes
:����������
conv_lstm2d_3/while/NoOpNoOp)^conv_lstm2d_3/while/split/ReadVariableOp+^conv_lstm2d_3/while/split_1/ReadVariableOp+^conv_lstm2d_3/while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/conv_lstm2d_3_while_conv_lstm2d_3_strided_slice1conv_lstm2d_3_while_conv_lstm2d_3_strided_slice_0"E
conv_lstm2d_3_while_identity%conv_lstm2d_3/while/Identity:output:0"I
conv_lstm2d_3_while_identity_1'conv_lstm2d_3/while/Identity_1:output:0"I
conv_lstm2d_3_while_identity_2'conv_lstm2d_3/while/Identity_2:output:0"I
conv_lstm2d_3_while_identity_3'conv_lstm2d_3/while/Identity_3:output:0"I
conv_lstm2d_3_while_identity_4'conv_lstm2d_3/while/Identity_4:output:0"I
conv_lstm2d_3_while_identity_5'conv_lstm2d_3/while/Identity_5:output:0"l
3conv_lstm2d_3_while_split_1_readvariableop_resource5conv_lstm2d_3_while_split_1_readvariableop_resource_0"l
3conv_lstm2d_3_while_split_2_readvariableop_resource5conv_lstm2d_3_while_split_2_readvariableop_resource_0"h
1conv_lstm2d_3_while_split_readvariableop_resource3conv_lstm2d_3_while_split_readvariableop_resource_0"�
mconv_lstm2d_3_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_3_tensorarrayunstack_tensorlistfromtensoroconv_lstm2d_3_while_tensorarrayv2read_tensorlistgetitem_conv_lstm2d_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :���������:���������: : : : : 2T
(conv_lstm2d_3/while/split/ReadVariableOp(conv_lstm2d_3/while/split/ReadVariableOp2X
*conv_lstm2d_3/while/split_1/ReadVariableOp*conv_lstm2d_3/while/split_1/ReadVariableOp2X
*conv_lstm2d_3/while/split_2/ReadVariableOp*conv_lstm2d_3/while/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�[
�
while_body_70013
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0@
%while_split_readvariableop_resource_0:�4A
'while_split_1_readvariableop_resource_0:45
'while_split_2_readvariableop_resource_0:4
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor>
#while_split_readvariableop_resource:�4?
%while_split_1_readvariableop_resource:43
%while_split_2_readvariableop_resource:4��while/split/ReadVariableOp�while/split_1/ReadVariableOp�while/split_2/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����        �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:����������*
element_dtype0W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:�4*
dtype0�
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*`
_output_shapesN
L:�:�:�:�*
	num_splitY
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
:4*
dtype0�
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes
:4*
dtype0�
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split�
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*/
_output_shapes
:����������
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*/
_output_shapes
:����������
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*/
_output_shapes
:����������
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*/
_output_shapes
:����������
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:���������P
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>R
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?o
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*/
_output_shapes
:���������u
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*/
_output_shapes
:���������b
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������Z
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:����������
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:���������R
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��L>R
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*/
_output_shapes
:���������w
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*/
_output_shapes
:���������d
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:���������\
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:���������|
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:����������
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:���������]

while/ReluReluwhile/add_4:z:0*
T0*/
_output_shapes
:���������
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*/
_output_shapes
:���������p
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:����������
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:���������R
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *��L>R
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*/
_output_shapes
:���������w
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*/
_output_shapes
:���������d
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:���������\
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:���������_
while/Relu_1Reluwhile/add_5:z:0*
T0*/
_output_shapes
:����������
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*/
_output_shapes
:���������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_5:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: O
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: [
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: t
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*/
_output_shapes
:���������t
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*/
_output_shapes
:����������

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :���������:���������: : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�	
�
#__inference_signature_wrapper_70354
reshape_3_input"
unknown:�4#
	unknown_0:4
	unknown_1:4
	unknown_2:	'�
	unknown_3:	�
	unknown_4:	�2
	unknown_5:2
	unknown_6:2
	unknown_7:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallreshape_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__wrapped_model_68972o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
,
_output_shapes
:����������
)
_user_specified_namereshape_3_input
�
F
*__inference_dropout_15_layer_call_fn_72061

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_15_layer_call_and_return_conditional_losses_69734`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�

�
C__inference_dense_10_layer_call_and_return_conditional_losses_69723

inputs1
matmul_readvariableop_resource:	�2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
E__inference_dropout_15_layer_call_and_return_conditional_losses_69734

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������2[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������2"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
c
E__inference_dropout_13_layer_call_and_return_conditional_losses_71966

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_69090
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_69090___redundant_placeholder03
/while_while_cond_69090___redundant_placeholder13
/while_while_cond_69090___redundant_placeholder23
/while_while_cond_69090___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :���������:���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_71114
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_71114___redundant_placeholder03
/while_while_cond_71114___redundant_placeholder13
/while_while_cond_71114___redundant_placeholder23
/while_while_cond_71114___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :���������:���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
(__inference_dense_10_layer_call_fn_72045

inputs
unknown:	�2
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_69723o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�=
�
K__inference_conv_lstm_cell_3_layer_call_and_return_conditional_losses_69076

inputs

states
states_18
split_readvariableop_resource:�49
split_1_readvariableop_resource:4-
split_2_readvariableop_resource:4
identity

identity_1

identity_2��split/ReadVariableOp�split_1/ReadVariableOp�split_2/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :{
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:�4*
dtype0�
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*`
_output_shapesN
L:�:�:�:�*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:4*
dtype0�
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
:4*
dtype0�
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split�
convolutionConv2Dinputssplit:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
t
BiasAddBiasAddconvolution:output:0split_2:output:0*
T0*/
_output_shapes
:����������
convolution_1Conv2Dinputssplit:output:1*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_1BiasAddconvolution_1:output:0split_2:output:1*
T0*/
_output_shapes
:����������
convolution_2Conv2Dinputssplit:output:2*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_2BiasAddconvolution_2:output:0split_2:output:2*
T0*/
_output_shapes
:����������
convolution_3Conv2Dinputssplit:output:3*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_3BiasAddconvolution_3:output:0split_2:output:3*
T0*/
_output_shapes
:����������
convolution_4Conv2Dstatessplit_1:output:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_5Conv2Dstatessplit_1:output:1*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_6Conv2Dstatessplit_1:output:2*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_7Conv2Dstatessplit_1:output:3*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_4:output:0*
T0*/
_output_shapes
:���������J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:���������c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:���������\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:���������t
add_2AddV2BiasAdd_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:���������L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:���������e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:���������^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:���������V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:���������e
mul_2Mulclip_by_value_1:z:0states_1*
T0*/
_output_shapes
:���������t
add_4AddV2BiasAdd_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:���������Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:���������m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:���������^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:���������t
add_6AddV2BiasAdd_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:���������L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:���������e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:���������^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:���������V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:���������S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:���������q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:���������`
IdentityIdentity	mul_5:z:0^NoOp*
T0*/
_output_shapes
:���������b

Identity_1Identity	mul_5:z:0^NoOp*
T0*/
_output_shapes
:���������b

Identity_2Identity	add_5:z:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:���������:���������: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_namestates:WS
/
_output_shapes
:���������
 
_user_specified_namestates
�	
d
E__inference_dropout_14_layer_call_and_return_conditional_losses_69838

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_dense_9_layer_call_fn_71998

inputs
unknown:	'�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_69699p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������': : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������'
 
_user_specified_nameinputs
�

�
B__inference_dense_9_layer_call_and_return_conditional_losses_69699

inputs1
matmul_readvariableop_resource:	'�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	'�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������': : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������'
 
_user_specified_nameinputs
�
�
while_cond_71786
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_71786___redundant_placeholder03
/while_while_cond_71786___redundant_placeholder13
/while_while_cond_71786___redundant_placeholder23
/while_while_cond_71786___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :���������:���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
c
*__inference_dropout_14_layer_call_fn_72019

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_14_layer_call_and_return_conditional_losses_69838p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�3
�
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_69160

inputs"
unknown:�4#
	unknown_0:4
	unknown_1:4
identity��StatefulPartitionedCall�whileg

zeros_like	ZerosLikeinputs*
T0*=
_output_shapes+
):'�������������������W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :u
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*0
_output_shapes
:����������n
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"           P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    u
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
k
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'�������������������B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����        �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:����������*
shrink_axis_mask�
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0convolution:output:0convolution:output:0unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:���������:���������:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_conv_lstm_cell_3_layer_call_and_return_conditional_losses_69076v
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0unknown	unknown_0	unknown_1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_69091*
condR
while_cond_69090*[
output_shapesJ
H: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:���������*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������*
shrink_axis_maskm
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:���������o
IdentityIdentitystrided_slice_2:output:0^NoOp*
T0*/
_output_shapes
:���������h
NoOpNoOp^StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'�������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:e a
=
_output_shapes+
):'�������������������
 
_user_specified_nameinputs
�
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_71989

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����'   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������'X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������'"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�e
�
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_70140

inputs8
split_readvariableop_resource:�49
split_1_readvariableop_resource:4-
split_2_readvariableop_resource:4
identity��split/ReadVariableOp�split_1/ReadVariableOp�split_2/ReadVariableOp�while^

zeros_like	ZerosLikeinputs*
T0*4
_output_shapes"
 :����������W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :u
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*0
_output_shapes
:����������n
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"           P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    u
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
k
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :����������B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����        �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:����������*
shrink_axis_maskQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :{
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:�4*
dtype0�
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*`
_output_shapesN
L:�:�:�:�*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:4*
dtype0�
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
:4*
dtype0�
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split�
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
v
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*/
_output_shapes
:����������
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*/
_output_shapes
:����������
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*/
_output_shapes
:����������
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*/
_output_shapes
:����������
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*/
_output_shapes
:���������J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:���������c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:���������\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:���������t
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*/
_output_shapes
:���������L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:���������e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:���������^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:���������V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:���������q
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:���������t
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*/
_output_shapes
:���������Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:���������m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:���������^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:���������t
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*/
_output_shapes
:���������L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:���������e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:���������^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:���������V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:���������S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:���������q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:���������v
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_70013*
condR
while_cond_70012*[
output_shapesJ
H: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:���������*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������*
shrink_axis_maskm
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:���������o
IdentityIdentitystrided_slice_2:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:����������: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�
�
while_cond_70012
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_70012___redundant_placeholder03
/while_while_cond_70012___redundant_placeholder13
/while_while_cond_70012___redundant_placeholder23
/while_while_cond_70012___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :���������:���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
conv_lstm2d_3_while_cond_705078
4conv_lstm2d_3_while_conv_lstm2d_3_while_loop_counter>
:conv_lstm2d_3_while_conv_lstm2d_3_while_maximum_iterations#
conv_lstm2d_3_while_placeholder%
!conv_lstm2d_3_while_placeholder_1%
!conv_lstm2d_3_while_placeholder_2%
!conv_lstm2d_3_while_placeholder_38
4conv_lstm2d_3_while_less_conv_lstm2d_3_strided_sliceO
Kconv_lstm2d_3_while_conv_lstm2d_3_while_cond_70507___redundant_placeholder0O
Kconv_lstm2d_3_while_conv_lstm2d_3_while_cond_70507___redundant_placeholder1O
Kconv_lstm2d_3_while_conv_lstm2d_3_while_cond_70507___redundant_placeholder2O
Kconv_lstm2d_3_while_conv_lstm2d_3_while_cond_70507___redundant_placeholder3 
conv_lstm2d_3_while_identity
�
conv_lstm2d_3/while/LessLessconv_lstm2d_3_while_placeholder4conv_lstm2d_3_while_less_conv_lstm2d_3_strided_slice*
T0*
_output_shapes
: g
conv_lstm2d_3/while/IdentityIdentityconv_lstm2d_3/while/Less:z:0*
T0
*
_output_shapes
: "E
conv_lstm2d_3_while_identity%conv_lstm2d_3/while/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :���������:���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_71951

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�=
�
K__inference_conv_lstm_cell_3_layer_call_and_return_conditional_losses_72212

inputs
states_0
states_18
split_readvariableop_resource:�49
split_1_readvariableop_resource:4-
split_2_readvariableop_resource:4
identity

identity_1

identity_2��split/ReadVariableOp�split_1/ReadVariableOp�split_2/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :{
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:�4*
dtype0�
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*`
_output_shapesN
L:�:�:�:�*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:4*
dtype0�
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
:4*
dtype0�
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split�
convolutionConv2Dinputssplit:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
t
BiasAddBiasAddconvolution:output:0split_2:output:0*
T0*/
_output_shapes
:����������
convolution_1Conv2Dinputssplit:output:1*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_1BiasAddconvolution_1:output:0split_2:output:1*
T0*/
_output_shapes
:����������
convolution_2Conv2Dinputssplit:output:2*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_2BiasAddconvolution_2:output:0split_2:output:2*
T0*/
_output_shapes
:����������
convolution_3Conv2Dinputssplit:output:3*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_3BiasAddconvolution_3:output:0split_2:output:3*
T0*/
_output_shapes
:����������
convolution_4Conv2Dstates_0split_1:output:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_5Conv2Dstates_0split_1:output:1*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_6Conv2Dstates_0split_1:output:2*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_7Conv2Dstates_0split_1:output:3*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_4:output:0*
T0*/
_output_shapes
:���������J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:���������c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:���������\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:���������t
add_2AddV2BiasAdd_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:���������L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:���������e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:���������^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:���������V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:���������e
mul_2Mulclip_by_value_1:z:0states_1*
T0*/
_output_shapes
:���������t
add_4AddV2BiasAdd_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:���������Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:���������m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:���������^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:���������t
add_6AddV2BiasAdd_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:���������L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:���������e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:���������^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:���������V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:���������S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:���������q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:���������`
IdentityIdentity	mul_5:z:0^NoOp*
T0*/
_output_shapes
:���������b

Identity_1Identity	mul_5:z:0^NoOp*
T0*/
_output_shapes
:���������b

Identity_2Identity	add_5:z:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:���������:���������: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:YU
/
_output_shapes
:���������
"
_user_specified_name
states/0:YU
/
_output_shapes
:���������
"
_user_specified_name
states/1
�
`
D__inference_reshape_3_layer_call_and_return_conditional_losses_69432

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :R
Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value
B :��
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:q
ReshapeReshapeinputsReshape/shape:output:0*
T0*4
_output_shapes"
 :����������e
IdentityIdentityReshape:output:0*
T0*4
_output_shapes"
 :����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
,__inference_sequential_3_layer_call_fn_69775
reshape_3_input"
unknown:�4#
	unknown_0:4
	unknown_1:4
	unknown_2:	'�
	unknown_3:	�
	unknown_4:	�2
	unknown_5:2
	unknown_6:2
	unknown_7:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallreshape_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_69754o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
,
_output_shapes
:����������
)
_user_specified_namereshape_3_input
�
c
*__inference_dropout_15_layer_call_fn_72066

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_15_layer_call_and_return_conditional_losses_69805o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������222
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
0__inference_conv_lstm_cell_3_layer_call_fn_72120

inputs
states_0
states_1"
unknown:�4#
	unknown_0:4
	unknown_1:4
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:���������:���������:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_conv_lstm_cell_3_layer_call_and_return_conditional_losses_69076w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:���������y

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:���������:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:YU
/
_output_shapes
:���������
"
_user_specified_name
states/0:YU
/
_output_shapes
:���������
"
_user_specified_name
states/1
��
�
G__inference_sequential_3_layer_call_and_return_conditional_losses_70954

inputsF
+conv_lstm2d_3_split_readvariableop_resource:�4G
-conv_lstm2d_3_split_1_readvariableop_resource:4;
-conv_lstm2d_3_split_2_readvariableop_resource:49
&dense_9_matmul_readvariableop_resource:	'�6
'dense_9_biasadd_readvariableop_resource:	�:
'dense_10_matmul_readvariableop_resource:	�26
(dense_10_biasadd_readvariableop_resource:29
'dense_11_matmul_readvariableop_resource:26
(dense_11_biasadd_readvariableop_resource:
identity��"conv_lstm2d_3/split/ReadVariableOp�$conv_lstm2d_3/split_1/ReadVariableOp�$conv_lstm2d_3/split_2/ReadVariableOp�conv_lstm2d_3/while�dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOpE
reshape_3/ShapeShapeinputs*
T0*
_output_shapes
:g
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_3/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value
B :��
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0"reshape_3/Reshape/shape/2:output:0"reshape_3/Reshape/shape/3:output:0"reshape_3/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:�
reshape_3/ReshapeReshapeinputs reshape_3/Reshape/shape:output:0*
T0*4
_output_shapes"
 :�����������
conv_lstm2d_3/zeros_like	ZerosLikereshape_3/Reshape:output:0*
T0*4
_output_shapes"
 :����������e
#conv_lstm2d_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
conv_lstm2d_3/SumSumconv_lstm2d_3/zeros_like:y:0,conv_lstm2d_3/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:����������|
#conv_lstm2d_3/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"           ^
conv_lstm2d_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv_lstm2d_3/zerosFill,conv_lstm2d_3/zeros/shape_as_tensor:output:0"conv_lstm2d_3/zeros/Const:output:0*
T0*'
_output_shapes
:��
conv_lstm2d_3/convolutionConv2Dconv_lstm2d_3/Sum:output:0conv_lstm2d_3/zeros:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
y
conv_lstm2d_3/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                �
conv_lstm2d_3/transpose	Transposereshape_3/Reshape:output:0%conv_lstm2d_3/transpose/perm:output:0*
T0*4
_output_shapes"
 :����������^
conv_lstm2d_3/ShapeShapeconv_lstm2d_3/transpose:y:0*
T0*
_output_shapes
:k
!conv_lstm2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#conv_lstm2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#conv_lstm2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv_lstm2d_3/strided_sliceStridedSliceconv_lstm2d_3/Shape:output:0*conv_lstm2d_3/strided_slice/stack:output:0,conv_lstm2d_3/strided_slice/stack_1:output:0,conv_lstm2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)conv_lstm2d_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv_lstm2d_3/TensorArrayV2TensorListReserve2conv_lstm2d_3/TensorArrayV2/element_shape:output:0$conv_lstm2d_3/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Cconv_lstm2d_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����        �
5conv_lstm2d_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorconv_lstm2d_3/transpose:y:0Lconv_lstm2d_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���m
#conv_lstm2d_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%conv_lstm2d_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%conv_lstm2d_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv_lstm2d_3/strided_slice_1StridedSliceconv_lstm2d_3/transpose:y:0,conv_lstm2d_3/strided_slice_1/stack:output:0.conv_lstm2d_3/strided_slice_1/stack_1:output:0.conv_lstm2d_3/strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:����������*
shrink_axis_mask_
conv_lstm2d_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
"conv_lstm2d_3/split/ReadVariableOpReadVariableOp+conv_lstm2d_3_split_readvariableop_resource*'
_output_shapes
:�4*
dtype0�
conv_lstm2d_3/splitSplit&conv_lstm2d_3/split/split_dim:output:0*conv_lstm2d_3/split/ReadVariableOp:value:0*
T0*`
_output_shapesN
L:�:�:�:�*
	num_splita
conv_lstm2d_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
$conv_lstm2d_3/split_1/ReadVariableOpReadVariableOp-conv_lstm2d_3_split_1_readvariableop_resource*&
_output_shapes
:4*
dtype0�
conv_lstm2d_3/split_1Split(conv_lstm2d_3/split_1/split_dim:output:0,conv_lstm2d_3/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splita
conv_lstm2d_3/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
$conv_lstm2d_3/split_2/ReadVariableOpReadVariableOp-conv_lstm2d_3_split_2_readvariableop_resource*
_output_shapes
:4*
dtype0�
conv_lstm2d_3/split_2Split(conv_lstm2d_3/split_2/split_dim:output:0,conv_lstm2d_3/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split�
conv_lstm2d_3/convolution_1Conv2D&conv_lstm2d_3/strided_slice_1:output:0conv_lstm2d_3/split:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv_lstm2d_3/BiasAddBiasAdd$conv_lstm2d_3/convolution_1:output:0conv_lstm2d_3/split_2:output:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/convolution_2Conv2D&conv_lstm2d_3/strided_slice_1:output:0conv_lstm2d_3/split:output:1*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv_lstm2d_3/BiasAdd_1BiasAdd$conv_lstm2d_3/convolution_2:output:0conv_lstm2d_3/split_2:output:1*
T0*/
_output_shapes
:����������
conv_lstm2d_3/convolution_3Conv2D&conv_lstm2d_3/strided_slice_1:output:0conv_lstm2d_3/split:output:2*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv_lstm2d_3/BiasAdd_2BiasAdd$conv_lstm2d_3/convolution_3:output:0conv_lstm2d_3/split_2:output:2*
T0*/
_output_shapes
:����������
conv_lstm2d_3/convolution_4Conv2D&conv_lstm2d_3/strided_slice_1:output:0conv_lstm2d_3/split:output:3*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv_lstm2d_3/BiasAdd_3BiasAdd$conv_lstm2d_3/convolution_4:output:0conv_lstm2d_3/split_2:output:3*
T0*/
_output_shapes
:����������
conv_lstm2d_3/convolution_5Conv2D"conv_lstm2d_3/convolution:output:0conv_lstm2d_3/split_1:output:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
conv_lstm2d_3/convolution_6Conv2D"conv_lstm2d_3/convolution:output:0conv_lstm2d_3/split_1:output:1*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
conv_lstm2d_3/convolution_7Conv2D"conv_lstm2d_3/convolution:output:0conv_lstm2d_3/split_1:output:2*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
conv_lstm2d_3/convolution_8Conv2D"conv_lstm2d_3/convolution:output:0conv_lstm2d_3/split_1:output:3*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
conv_lstm2d_3/addAddV2conv_lstm2d_3/BiasAdd:output:0$conv_lstm2d_3/convolution_5:output:0*
T0*/
_output_shapes
:���������X
conv_lstm2d_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>Z
conv_lstm2d_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?�
conv_lstm2d_3/MulMulconv_lstm2d_3/add:z:0conv_lstm2d_3/Const:output:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/Add_1AddV2conv_lstm2d_3/Mul:z:0conv_lstm2d_3/Const_1:output:0*
T0*/
_output_shapes
:���������j
%conv_lstm2d_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
#conv_lstm2d_3/clip_by_value/MinimumMinimumconv_lstm2d_3/Add_1:z:0.conv_lstm2d_3/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������b
conv_lstm2d_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv_lstm2d_3/clip_by_valueMaximum'conv_lstm2d_3/clip_by_value/Minimum:z:0&conv_lstm2d_3/clip_by_value/y:output:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/add_2AddV2 conv_lstm2d_3/BiasAdd_1:output:0$conv_lstm2d_3/convolution_6:output:0*
T0*/
_output_shapes
:���������Z
conv_lstm2d_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��L>Z
conv_lstm2d_3/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?�
conv_lstm2d_3/Mul_1Mulconv_lstm2d_3/add_2:z:0conv_lstm2d_3/Const_2:output:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/Add_3AddV2conv_lstm2d_3/Mul_1:z:0conv_lstm2d_3/Const_3:output:0*
T0*/
_output_shapes
:���������l
'conv_lstm2d_3/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%conv_lstm2d_3/clip_by_value_1/MinimumMinimumconv_lstm2d_3/Add_3:z:00conv_lstm2d_3/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:���������d
conv_lstm2d_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv_lstm2d_3/clip_by_value_1Maximum)conv_lstm2d_3/clip_by_value_1/Minimum:z:0(conv_lstm2d_3/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/mul_2Mul!conv_lstm2d_3/clip_by_value_1:z:0"conv_lstm2d_3/convolution:output:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/add_4AddV2 conv_lstm2d_3/BiasAdd_2:output:0$conv_lstm2d_3/convolution_7:output:0*
T0*/
_output_shapes
:���������m
conv_lstm2d_3/ReluReluconv_lstm2d_3/add_4:z:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/mul_3Mulconv_lstm2d_3/clip_by_value:z:0 conv_lstm2d_3/Relu:activations:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/add_5AddV2conv_lstm2d_3/mul_2:z:0conv_lstm2d_3/mul_3:z:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/add_6AddV2 conv_lstm2d_3/BiasAdd_3:output:0$conv_lstm2d_3/convolution_8:output:0*
T0*/
_output_shapes
:���������Z
conv_lstm2d_3/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *��L>Z
conv_lstm2d_3/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?�
conv_lstm2d_3/Mul_4Mulconv_lstm2d_3/add_6:z:0conv_lstm2d_3/Const_4:output:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/Add_7AddV2conv_lstm2d_3/Mul_4:z:0conv_lstm2d_3/Const_5:output:0*
T0*/
_output_shapes
:���������l
'conv_lstm2d_3/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%conv_lstm2d_3/clip_by_value_2/MinimumMinimumconv_lstm2d_3/Add_7:z:00conv_lstm2d_3/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:���������d
conv_lstm2d_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv_lstm2d_3/clip_by_value_2Maximum)conv_lstm2d_3/clip_by_value_2/Minimum:z:0(conv_lstm2d_3/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:���������o
conv_lstm2d_3/Relu_1Reluconv_lstm2d_3/add_5:z:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/mul_5Mul!conv_lstm2d_3/clip_by_value_2:z:0"conv_lstm2d_3/Relu_1:activations:0*
T0*/
_output_shapes
:����������
+conv_lstm2d_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
*conv_lstm2d_3/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
conv_lstm2d_3/TensorArrayV2_1TensorListReserve4conv_lstm2d_3/TensorArrayV2_1/element_shape:output:03conv_lstm2d_3/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���T
conv_lstm2d_3/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&conv_lstm2d_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������b
 conv_lstm2d_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
conv_lstm2d_3/whileWhile)conv_lstm2d_3/while/loop_counter:output:0/conv_lstm2d_3/while/maximum_iterations:output:0conv_lstm2d_3/time:output:0&conv_lstm2d_3/TensorArrayV2_1:handle:0"conv_lstm2d_3/convolution:output:0"conv_lstm2d_3/convolution:output:0$conv_lstm2d_3/strided_slice:output:0Econv_lstm2d_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0+conv_lstm2d_3_split_readvariableop_resource-conv_lstm2d_3_split_1_readvariableop_resource-conv_lstm2d_3_split_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( **
body"R 
conv_lstm2d_3_while_body_70771**
cond"R 
conv_lstm2d_3_while_cond_70770*[
output_shapesJ
H: : : : :���������:���������: : : : : *
parallel_iterations �
>conv_lstm2d_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
0conv_lstm2d_3/TensorArrayV2Stack/TensorListStackTensorListStackconv_lstm2d_3/while:output:3Gconv_lstm2d_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:���������*
element_dtype0*
num_elementsv
#conv_lstm2d_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������o
%conv_lstm2d_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%conv_lstm2d_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv_lstm2d_3/strided_slice_2StridedSlice9conv_lstm2d_3/TensorArrayV2Stack/TensorListStack:tensor:0,conv_lstm2d_3/strided_slice_2/stack:output:0.conv_lstm2d_3/strided_slice_2/stack_1:output:0.conv_lstm2d_3/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������*
shrink_axis_mask{
conv_lstm2d_3/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                �
conv_lstm2d_3/transpose_1	Transpose9conv_lstm2d_3/TensorArrayV2Stack/TensorListStack:tensor:0'conv_lstm2d_3/transpose_1/perm:output:0*
T0*3
_output_shapes!
:���������]
dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_12/dropout/MulMul&conv_lstm2d_3/strided_slice_2:output:0!dropout_12/dropout/Const:output:0*
T0*/
_output_shapes
:���������n
dropout_12/dropout/ShapeShape&conv_lstm2d_3/strided_slice_2:output:0*
T0*
_output_shapes
:�
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*/
_output_shapes
:���������*
dtype0f
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:����������
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:����������
dropout_12/dropout/Mul_1Muldropout_12/dropout/Mul:z:0dropout_12/dropout/Cast:y:0*
T0*/
_output_shapes
:����������
max_pooling2d_3/MaxPoolMaxPooldropout_12/dropout/Mul_1:z:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
]
dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_13/dropout/MulMul max_pooling2d_3/MaxPool:output:0!dropout_13/dropout/Const:output:0*
T0*/
_output_shapes
:���������h
dropout_13/dropout/ShapeShape max_pooling2d_3/MaxPool:output:0*
T0*
_output_shapes
:�
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*/
_output_shapes
:���������*
dtype0f
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:����������
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:����������
dropout_13/dropout/Mul_1Muldropout_13/dropout/Mul:z:0dropout_13/dropout/Cast:y:0*
T0*/
_output_shapes
:���������`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����'   �
flatten_3/ReshapeReshapedropout_13/dropout/Mul_1:z:0flatten_3/Const:output:0*
T0*'
_output_shapes
:���������'�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	'�*
dtype0�
dense_9/MatMulMatMulflatten_3/Reshape:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:����������]
dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_14/dropout/MulMuldense_9/Relu:activations:0!dropout_14/dropout/Const:output:0*
T0*(
_output_shapes
:����������b
dropout_14/dropout/ShapeShapedense_9/Relu:activations:0*
T0*
_output_shapes
:�
/dropout_14/dropout/random_uniform/RandomUniformRandomUniform!dropout_14/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0f
!dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_14/dropout/GreaterEqualGreaterEqual8dropout_14/dropout/random_uniform/RandomUniform:output:0*dropout_14/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:�����������
dropout_14/dropout/CastCast#dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:�����������
dropout_14/dropout/Mul_1Muldropout_14/dropout/Mul:z:0dropout_14/dropout/Cast:y:0*
T0*(
_output_shapes
:�����������
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes
:	�2*
dtype0�
dense_10/MatMulMatMuldropout_14/dropout/Mul_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2b
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:���������2]
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_15/dropout/MulMuldense_10/Relu:activations:0!dropout_15/dropout/Const:output:0*
T0*'
_output_shapes
:���������2c
dropout_15/dropout/ShapeShapedense_10/Relu:activations:0*
T0*
_output_shapes
:�
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*'
_output_shapes
:���������2*
dtype0f
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������2�
dropout_15/dropout/CastCast#dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������2�
dropout_15/dropout/Mul_1Muldropout_15/dropout/Mul:z:0dropout_15/dropout/Cast:y:0*
T0*'
_output_shapes
:���������2�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
dense_11/MatMulMatMuldropout_15/dropout/Mul_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_11/SigmoidSigmoiddense_11/BiasAdd:output:0*
T0*'
_output_shapes
:���������c
IdentityIdentitydense_11/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^conv_lstm2d_3/split/ReadVariableOp%^conv_lstm2d_3/split_1/ReadVariableOp%^conv_lstm2d_3/split_2/ReadVariableOp^conv_lstm2d_3/while ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 2H
"conv_lstm2d_3/split/ReadVariableOp"conv_lstm2d_3/split/ReadVariableOp2L
$conv_lstm2d_3/split_1/ReadVariableOp$conv_lstm2d_3/split_1/ReadVariableOp2L
$conv_lstm2d_3/split_2/ReadVariableOp$conv_lstm2d_3/split_2/ReadVariableOp2*
conv_lstm2d_3/whileconv_lstm2d_3/while2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+sequential_3_conv_lstm2d_3_while_cond_68816R
Nsequential_3_conv_lstm2d_3_while_sequential_3_conv_lstm2d_3_while_loop_counterX
Tsequential_3_conv_lstm2d_3_while_sequential_3_conv_lstm2d_3_while_maximum_iterations0
,sequential_3_conv_lstm2d_3_while_placeholder2
.sequential_3_conv_lstm2d_3_while_placeholder_12
.sequential_3_conv_lstm2d_3_while_placeholder_22
.sequential_3_conv_lstm2d_3_while_placeholder_3R
Nsequential_3_conv_lstm2d_3_while_less_sequential_3_conv_lstm2d_3_strided_slicei
esequential_3_conv_lstm2d_3_while_sequential_3_conv_lstm2d_3_while_cond_68816___redundant_placeholder0i
esequential_3_conv_lstm2d_3_while_sequential_3_conv_lstm2d_3_while_cond_68816___redundant_placeholder1i
esequential_3_conv_lstm2d_3_while_sequential_3_conv_lstm2d_3_while_cond_68816___redundant_placeholder2i
esequential_3_conv_lstm2d_3_while_sequential_3_conv_lstm2d_3_while_cond_68816___redundant_placeholder3-
)sequential_3_conv_lstm2d_3_while_identity
�
%sequential_3/conv_lstm2d_3/while/LessLess,sequential_3_conv_lstm2d_3_while_placeholderNsequential_3_conv_lstm2d_3_while_less_sequential_3_conv_lstm2d_3_strided_slice*
T0*
_output_shapes
: �
)sequential_3/conv_lstm2d_3/while/IdentityIdentity)sequential_3/conv_lstm2d_3/while/Less:z:0*
T0
*
_output_shapes
: "_
)sequential_3_conv_lstm2d_3_while_identity2sequential_3/conv_lstm2d_3/while/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :���������:���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�

�
B__inference_dense_9_layer_call_and_return_conditional_losses_72009

inputs1
matmul_readvariableop_resource:	'�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	'�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������': : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������'
 
_user_specified_nameinputs
�
c
E__inference_dropout_14_layer_call_and_return_conditional_losses_72024

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�+
�
G__inference_sequential_3_layer_call_and_return_conditional_losses_70290
reshape_3_input.
conv_lstm2d_3_70261:�4-
conv_lstm2d_3_70263:4!
conv_lstm2d_3_70265:4 
dense_9_70272:	'�
dense_9_70274:	�!
dense_10_70278:	�2
dense_10_70280:2 
dense_11_70284:2
dense_11_70286:
identity��%conv_lstm2d_3/StatefulPartitionedCall� dense_10/StatefulPartitionedCall� dense_11/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�
reshape_3/PartitionedCallPartitionedCallreshape_3_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_69432�
%conv_lstm2d_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0conv_lstm2d_3_70261conv_lstm2d_3_70263conv_lstm2d_3_70265*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_69657�
dropout_12/PartitionedCallPartitionedCall.conv_lstm2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_12_layer_call_and_return_conditional_losses_69670�
max_pooling2d_3/PartitionedCallPartitionedCall#dropout_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_69407�
dropout_13/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_13_layer_call_and_return_conditional_losses_69678�
flatten_3/PartitionedCallPartitionedCall#dropout_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������'* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_69686�
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_9_70272dense_9_70274*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_69699�
dropout_14/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_14_layer_call_and_return_conditional_losses_69710�
 dense_10/StatefulPartitionedCallStatefulPartitionedCall#dropout_14/PartitionedCall:output:0dense_10_70278dense_10_70280*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_69723�
dropout_15/PartitionedCallPartitionedCall)dense_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_15_layer_call_and_return_conditional_losses_69734�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall#dropout_15/PartitionedCall:output:0dense_11_70284dense_11_70286*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_69747x
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp&^conv_lstm2d_3/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 2N
%conv_lstm2d_3/StatefulPartitionedCall%conv_lstm2d_3/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:] Y
,
_output_shapes
:����������
)
_user_specified_namereshape_3_input
�
E
)__inference_flatten_3_layer_call_fn_71983

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������'* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_69686`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������'"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
C__inference_dense_11_layer_call_and_return_conditional_losses_69747

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�

�
C__inference_dense_11_layer_call_and_return_conditional_losses_72103

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�

d
E__inference_dropout_13_layer_call_and_return_conditional_losses_71978

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�+
�
G__inference_sequential_3_layer_call_and_return_conditional_losses_69754

inputs.
conv_lstm2d_3_69658:�4-
conv_lstm2d_3_69660:4!
conv_lstm2d_3_69662:4 
dense_9_69700:	'�
dense_9_69702:	�!
dense_10_69724:	�2
dense_10_69726:2 
dense_11_69748:2
dense_11_69750:
identity��%conv_lstm2d_3/StatefulPartitionedCall� dense_10/StatefulPartitionedCall� dense_11/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�
reshape_3/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_69432�
%conv_lstm2d_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0conv_lstm2d_3_69658conv_lstm2d_3_69660conv_lstm2d_3_69662*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_69657�
dropout_12/PartitionedCallPartitionedCall.conv_lstm2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_12_layer_call_and_return_conditional_losses_69670�
max_pooling2d_3/PartitionedCallPartitionedCall#dropout_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_69407�
dropout_13/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_13_layer_call_and_return_conditional_losses_69678�
flatten_3/PartitionedCallPartitionedCall#dropout_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������'* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_69686�
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_9_69700dense_9_69702*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_69699�
dropout_14/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_14_layer_call_and_return_conditional_losses_69710�
 dense_10/StatefulPartitionedCallStatefulPartitionedCall#dropout_14/PartitionedCall:output:0dense_10_69724dense_10_69726*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_69723�
dropout_15/PartitionedCallPartitionedCall)dense_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_15_layer_call_and_return_conditional_losses_69734�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall#dropout_15/PartitionedCall:output:0dense_11_69748dense_11_69750*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_69747x
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp&^conv_lstm2d_3/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 2N
%conv_lstm2d_3/StatefulPartitionedCall%conv_lstm2d_3/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
F
*__inference_dropout_13_layer_call_fn_71956

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_13_layer_call_and_return_conditional_losses_69678h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
,__inference_sequential_3_layer_call_fn_70400

inputs"
unknown:�4#
	unknown_0:4
	unknown_1:4
	unknown_2:	'�
	unknown_3:	�
	unknown_4:	�2
	unknown_5:2
	unknown_6:2
	unknown_7:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_70213o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
F
*__inference_dropout_14_layer_call_fn_72014

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_14_layer_call_and_return_conditional_losses_69710a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�e
�
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_69657

inputs8
split_readvariableop_resource:�49
split_1_readvariableop_resource:4-
split_2_readvariableop_resource:4
identity��split/ReadVariableOp�split_1/ReadVariableOp�split_2/ReadVariableOp�while^

zeros_like	ZerosLikeinputs*
T0*4
_output_shapes"
 :����������W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :u
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*0
_output_shapes
:����������n
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"           P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    u
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
k
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :����������B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����        �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:����������*
shrink_axis_maskQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :{
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:�4*
dtype0�
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*`
_output_shapesN
L:�:�:�:�*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:4*
dtype0�
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes
:4*
dtype0�
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split�
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
v
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*/
_output_shapes
:����������
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*/
_output_shapes
:����������
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*/
_output_shapes
:����������
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
x
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*/
_output_shapes
:����������
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
p
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*/
_output_shapes
:���������J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:���������c
Add_1AddV2Mul:z:0Const_1:output:0*
T0*/
_output_shapes
:���������\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:���������t
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*/
_output_shapes
:���������L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:���������e
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:���������^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:���������V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:���������q
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:���������t
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*/
_output_shapes
:���������Q
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:���������m
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:���������^
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:���������t
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*/
_output_shapes
:���������L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?c
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:���������e
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:���������^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:���������V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:���������S
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:���������q
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:���������v
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_69530*
condR
while_cond_69529*[
output_shapesJ
H: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:���������*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������*
shrink_axis_maskm
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:���������o
IdentityIdentitystrided_slice_2:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:����������: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�

�
,__inference_sequential_3_layer_call_fn_70377

inputs"
unknown:�4#
	unknown_0:4
	unknown_1:4
	unknown_2:	'�
	unknown_3:	�
	unknown_4:	�2
	unknown_5:2
	unknown_6:2
	unknown_7:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_69754o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�L
�
__inference__traced_save_72418
file_prefix-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop3
/savev2_conv_lstm2d_3_kernel_read_readvariableop=
9savev2_conv_lstm2d_3_recurrent_kernel_read_readvariableop1
-savev2_conv_lstm2d_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableop5
1savev2_adam_dense_10_kernel_m_read_readvariableop3
/savev2_adam_dense_10_bias_m_read_readvariableop5
1savev2_adam_dense_11_kernel_m_read_readvariableop3
/savev2_adam_dense_11_bias_m_read_readvariableop:
6savev2_adam_conv_lstm2d_3_kernel_m_read_readvariableopD
@savev2_adam_conv_lstm2d_3_recurrent_kernel_m_read_readvariableop8
4savev2_adam_conv_lstm2d_3_bias_m_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableop5
1savev2_adam_dense_10_kernel_v_read_readvariableop3
/savev2_adam_dense_10_bias_v_read_readvariableop5
1savev2_adam_dense_11_kernel_v_read_readvariableop3
/savev2_adam_dense_11_bias_v_read_readvariableop:
6savev2_adam_conv_lstm2d_3_kernel_v_read_readvariableopD
@savev2_adam_conv_lstm2d_3_recurrent_kernel_v_read_readvariableop8
4savev2_adam_conv_lstm2d_3_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*�
value�B�%B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop/savev2_conv_lstm2d_3_kernel_read_readvariableop9savev2_conv_lstm2d_3_recurrent_kernel_read_readvariableop-savev2_conv_lstm2d_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableop1savev2_adam_dense_10_kernel_m_read_readvariableop/savev2_adam_dense_10_bias_m_read_readvariableop1savev2_adam_dense_11_kernel_m_read_readvariableop/savev2_adam_dense_11_bias_m_read_readvariableop6savev2_adam_conv_lstm2d_3_kernel_m_read_readvariableop@savev2_adam_conv_lstm2d_3_recurrent_kernel_m_read_readvariableop4savev2_adam_conv_lstm2d_3_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableop1savev2_adam_dense_10_kernel_v_read_readvariableop/savev2_adam_dense_10_bias_v_read_readvariableop1savev2_adam_dense_11_kernel_v_read_readvariableop/savev2_adam_dense_11_bias_v_read_readvariableop6savev2_adam_conv_lstm2d_3_kernel_v_read_readvariableop@savev2_adam_conv_lstm2d_3_recurrent_kernel_v_read_readvariableop4savev2_adam_conv_lstm2d_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *3
dtypes)
'2%	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	'�:�:	�2:2:2::�4:4:4: : : : : : : : : :	'�:�:	�2:2:2::�4:4:4:	'�:�:	�2:2:2::�4:4:4: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	'�:!

_output_shapes	
:�:%!

_output_shapes
:	�2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::-)
'
_output_shapes
:�4:,(
&
_output_shapes
:4: 	

_output_shapes
:4:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	'�:!

_output_shapes	
:�:%!

_output_shapes
:	�2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::-)
'
_output_shapes
:�4:,(
&
_output_shapes
:4: 

_output_shapes
:4:%!

_output_shapes
:	'�:!

_output_shapes	
:�:%!

_output_shapes
:	�2: 

_output_shapes
:2:$  

_output_shapes

:2: !

_output_shapes
::-")
'
_output_shapes
:�4:,#(
&
_output_shapes
:4: $

_output_shapes
:4:%

_output_shapes
: 
�
�
-__inference_conv_lstm2d_3_layer_call_fn_70996
inputs_0"
unknown:�4#
	unknown_0:4
	unknown_1:4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_69389w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'�������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'�������������������
"
_user_specified_name
inputs/0
�
f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_69407

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
��
�
!__inference__traced_restore_72536
file_prefix2
assignvariableop_dense_9_kernel:	'�.
assignvariableop_1_dense_9_bias:	�5
"assignvariableop_2_dense_10_kernel:	�2.
 assignvariableop_3_dense_10_bias:24
"assignvariableop_4_dense_11_kernel:2.
 assignvariableop_5_dense_11_bias:B
'assignvariableop_6_conv_lstm2d_3_kernel:�4K
1assignvariableop_7_conv_lstm2d_3_recurrent_kernel:43
%assignvariableop_8_conv_lstm2d_3_bias:4&
assignvariableop_9_adam_iter:	 )
assignvariableop_10_adam_beta_1: )
assignvariableop_11_adam_beta_2: (
assignvariableop_12_adam_decay: 0
&assignvariableop_13_adam_learning_rate: %
assignvariableop_14_total_1: %
assignvariableop_15_count_1: #
assignvariableop_16_total: #
assignvariableop_17_count: <
)assignvariableop_18_adam_dense_9_kernel_m:	'�6
'assignvariableop_19_adam_dense_9_bias_m:	�=
*assignvariableop_20_adam_dense_10_kernel_m:	�26
(assignvariableop_21_adam_dense_10_bias_m:2<
*assignvariableop_22_adam_dense_11_kernel_m:26
(assignvariableop_23_adam_dense_11_bias_m:J
/assignvariableop_24_adam_conv_lstm2d_3_kernel_m:�4S
9assignvariableop_25_adam_conv_lstm2d_3_recurrent_kernel_m:4;
-assignvariableop_26_adam_conv_lstm2d_3_bias_m:4<
)assignvariableop_27_adam_dense_9_kernel_v:	'�6
'assignvariableop_28_adam_dense_9_bias_v:	�=
*assignvariableop_29_adam_dense_10_kernel_v:	�26
(assignvariableop_30_adam_dense_10_bias_v:2<
*assignvariableop_31_adam_dense_11_kernel_v:26
(assignvariableop_32_adam_dense_11_bias_v:J
/assignvariableop_33_adam_conv_lstm2d_3_kernel_v:�4S
9assignvariableop_34_adam_conv_lstm2d_3_recurrent_kernel_v:4;
-assignvariableop_35_adam_conv_lstm2d_3_bias_v:4
identity_37��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*�
value�B�%B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_9_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_9_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_10_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_10_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_11_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_11_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp'assignvariableop_6_conv_lstm2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp1assignvariableop_7_conv_lstm2d_3_recurrent_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp%assignvariableop_8_conv_lstm2d_3_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_iterIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_2Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_decayIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp&assignvariableop_13_adam_learning_rateIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_dense_9_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_dense_9_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_10_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_dense_10_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_11_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_dense_11_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp/assignvariableop_24_adam_conv_lstm2d_3_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp9assignvariableop_25_adam_conv_lstm2d_3_recurrent_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp-assignvariableop_26_adam_conv_lstm2d_3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_9_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_9_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_10_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_10_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_11_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_11_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp/assignvariableop_33_adam_conv_lstm2d_3_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp9assignvariableop_34_adam_conv_lstm2d_3_recurrent_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp-assignvariableop_35_adam_conv_lstm2d_3_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_36Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_37IdentityIdentity_36:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_37Identity_37:output:0*]
_input_shapesL
J: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
-__inference_conv_lstm2d_3_layer_call_fn_70985
inputs_0"
unknown:�4#
	unknown_0:4
	unknown_1:4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_69160w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'�������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'�������������������
"
_user_specified_name
inputs/0
�	
d
E__inference_dropout_14_layer_call_and_return_conditional_losses_72036

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�[
�
while_body_71787
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0@
%while_split_readvariableop_resource_0:�4A
'while_split_1_readvariableop_resource_0:45
'while_split_2_readvariableop_resource_0:4
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor>
#while_split_readvariableop_resource:�4?
%while_split_1_readvariableop_resource:43
%while_split_2_readvariableop_resource:4��while/split/ReadVariableOp�while/split_1/ReadVariableOp�while/split_2/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����        �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:����������*
element_dtype0W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:�4*
dtype0�
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*`
_output_shapesN
L:�:�:�:�*
	num_splitY
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
:4*
dtype0�
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splitY
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes
:4*
dtype0�
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split�
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*/
_output_shapes
:����������
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*/
_output_shapes
:����������
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*/
_output_shapes
:����������
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*/
_output_shapes
:����������
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:���������P
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>R
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?o
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*/
_output_shapes
:���������u
while/Add_1AddV2while/Mul:z:0while/Const_1:output:0*
T0*/
_output_shapes
:���������b
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������Z
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:����������
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:���������R
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��L>R
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*/
_output_shapes
:���������w
while/Add_3AddV2while/Mul_1:z:0while/Const_3:output:0*
T0*/
_output_shapes
:���������d
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:���������\
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:���������|
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:����������
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:���������]

while/ReluReluwhile/add_4:z:0*
T0*/
_output_shapes
:���������
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*/
_output_shapes
:���������p
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:����������
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:���������R
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *��L>R
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?u
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*/
_output_shapes
:���������w
while/Add_7AddV2while/Mul_4:z:0while/Const_5:output:0*
T0*/
_output_shapes
:���������d
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:���������\
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:���������_
while/Relu_1Reluwhile/add_5:z:0*
T0*/
_output_shapes
:����������
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*/
_output_shapes
:���������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_5:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: O
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_9:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: [
while/Identity_2Identitywhile/add_8:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: t
while/Identity_4Identitywhile/mul_5:z:0^while/NoOp*
T0*/
_output_shapes
:���������t
while/Identity_5Identitywhile/add_5:z:0^while/NoOp*
T0*/
_output_shapes
:����������

while/NoOpNoOp^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :���������:���������: : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�	
d
E__inference_dropout_15_layer_call_and_return_conditional_losses_69805

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������2C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������2*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������2o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������2i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������2Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
c
E__inference_dropout_14_layer_call_and_return_conditional_losses_69710

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
G__inference_sequential_3_layer_call_and_return_conditional_losses_70663

inputsF
+conv_lstm2d_3_split_readvariableop_resource:�4G
-conv_lstm2d_3_split_1_readvariableop_resource:4;
-conv_lstm2d_3_split_2_readvariableop_resource:49
&dense_9_matmul_readvariableop_resource:	'�6
'dense_9_biasadd_readvariableop_resource:	�:
'dense_10_matmul_readvariableop_resource:	�26
(dense_10_biasadd_readvariableop_resource:29
'dense_11_matmul_readvariableop_resource:26
(dense_11_biasadd_readvariableop_resource:
identity��"conv_lstm2d_3/split/ReadVariableOp�$conv_lstm2d_3/split_1/ReadVariableOp�$conv_lstm2d_3/split_2/ReadVariableOp�conv_lstm2d_3/while�dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOpE
reshape_3/ShapeShapeinputs*
T0*
_output_shapes
:g
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_3/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value
B :��
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0"reshape_3/Reshape/shape/2:output:0"reshape_3/Reshape/shape/3:output:0"reshape_3/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:�
reshape_3/ReshapeReshapeinputs reshape_3/Reshape/shape:output:0*
T0*4
_output_shapes"
 :�����������
conv_lstm2d_3/zeros_like	ZerosLikereshape_3/Reshape:output:0*
T0*4
_output_shapes"
 :����������e
#conv_lstm2d_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
conv_lstm2d_3/SumSumconv_lstm2d_3/zeros_like:y:0,conv_lstm2d_3/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:����������|
#conv_lstm2d_3/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"           ^
conv_lstm2d_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv_lstm2d_3/zerosFill,conv_lstm2d_3/zeros/shape_as_tensor:output:0"conv_lstm2d_3/zeros/Const:output:0*
T0*'
_output_shapes
:��
conv_lstm2d_3/convolutionConv2Dconv_lstm2d_3/Sum:output:0conv_lstm2d_3/zeros:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
y
conv_lstm2d_3/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                �
conv_lstm2d_3/transpose	Transposereshape_3/Reshape:output:0%conv_lstm2d_3/transpose/perm:output:0*
T0*4
_output_shapes"
 :����������^
conv_lstm2d_3/ShapeShapeconv_lstm2d_3/transpose:y:0*
T0*
_output_shapes
:k
!conv_lstm2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#conv_lstm2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#conv_lstm2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv_lstm2d_3/strided_sliceStridedSliceconv_lstm2d_3/Shape:output:0*conv_lstm2d_3/strided_slice/stack:output:0,conv_lstm2d_3/strided_slice/stack_1:output:0,conv_lstm2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)conv_lstm2d_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv_lstm2d_3/TensorArrayV2TensorListReserve2conv_lstm2d_3/TensorArrayV2/element_shape:output:0$conv_lstm2d_3/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Cconv_lstm2d_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����        �
5conv_lstm2d_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorconv_lstm2d_3/transpose:y:0Lconv_lstm2d_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���m
#conv_lstm2d_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%conv_lstm2d_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%conv_lstm2d_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv_lstm2d_3/strided_slice_1StridedSliceconv_lstm2d_3/transpose:y:0,conv_lstm2d_3/strided_slice_1/stack:output:0.conv_lstm2d_3/strided_slice_1/stack_1:output:0.conv_lstm2d_3/strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:����������*
shrink_axis_mask_
conv_lstm2d_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
"conv_lstm2d_3/split/ReadVariableOpReadVariableOp+conv_lstm2d_3_split_readvariableop_resource*'
_output_shapes
:�4*
dtype0�
conv_lstm2d_3/splitSplit&conv_lstm2d_3/split/split_dim:output:0*conv_lstm2d_3/split/ReadVariableOp:value:0*
T0*`
_output_shapesN
L:�:�:�:�*
	num_splita
conv_lstm2d_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
$conv_lstm2d_3/split_1/ReadVariableOpReadVariableOp-conv_lstm2d_3_split_1_readvariableop_resource*&
_output_shapes
:4*
dtype0�
conv_lstm2d_3/split_1Split(conv_lstm2d_3/split_1/split_dim:output:0,conv_lstm2d_3/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_splita
conv_lstm2d_3/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
$conv_lstm2d_3/split_2/ReadVariableOpReadVariableOp-conv_lstm2d_3_split_2_readvariableop_resource*
_output_shapes
:4*
dtype0�
conv_lstm2d_3/split_2Split(conv_lstm2d_3/split_2/split_dim:output:0,conv_lstm2d_3/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split�
conv_lstm2d_3/convolution_1Conv2D&conv_lstm2d_3/strided_slice_1:output:0conv_lstm2d_3/split:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv_lstm2d_3/BiasAddBiasAdd$conv_lstm2d_3/convolution_1:output:0conv_lstm2d_3/split_2:output:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/convolution_2Conv2D&conv_lstm2d_3/strided_slice_1:output:0conv_lstm2d_3/split:output:1*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv_lstm2d_3/BiasAdd_1BiasAdd$conv_lstm2d_3/convolution_2:output:0conv_lstm2d_3/split_2:output:1*
T0*/
_output_shapes
:����������
conv_lstm2d_3/convolution_3Conv2D&conv_lstm2d_3/strided_slice_1:output:0conv_lstm2d_3/split:output:2*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv_lstm2d_3/BiasAdd_2BiasAdd$conv_lstm2d_3/convolution_3:output:0conv_lstm2d_3/split_2:output:2*
T0*/
_output_shapes
:����������
conv_lstm2d_3/convolution_4Conv2D&conv_lstm2d_3/strided_slice_1:output:0conv_lstm2d_3/split:output:3*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv_lstm2d_3/BiasAdd_3BiasAdd$conv_lstm2d_3/convolution_4:output:0conv_lstm2d_3/split_2:output:3*
T0*/
_output_shapes
:����������
conv_lstm2d_3/convolution_5Conv2D"conv_lstm2d_3/convolution:output:0conv_lstm2d_3/split_1:output:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
conv_lstm2d_3/convolution_6Conv2D"conv_lstm2d_3/convolution:output:0conv_lstm2d_3/split_1:output:1*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
conv_lstm2d_3/convolution_7Conv2D"conv_lstm2d_3/convolution:output:0conv_lstm2d_3/split_1:output:2*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
conv_lstm2d_3/convolution_8Conv2D"conv_lstm2d_3/convolution:output:0conv_lstm2d_3/split_1:output:3*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
conv_lstm2d_3/addAddV2conv_lstm2d_3/BiasAdd:output:0$conv_lstm2d_3/convolution_5:output:0*
T0*/
_output_shapes
:���������X
conv_lstm2d_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>Z
conv_lstm2d_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?�
conv_lstm2d_3/MulMulconv_lstm2d_3/add:z:0conv_lstm2d_3/Const:output:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/Add_1AddV2conv_lstm2d_3/Mul:z:0conv_lstm2d_3/Const_1:output:0*
T0*/
_output_shapes
:���������j
%conv_lstm2d_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
#conv_lstm2d_3/clip_by_value/MinimumMinimumconv_lstm2d_3/Add_1:z:0.conv_lstm2d_3/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������b
conv_lstm2d_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv_lstm2d_3/clip_by_valueMaximum'conv_lstm2d_3/clip_by_value/Minimum:z:0&conv_lstm2d_3/clip_by_value/y:output:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/add_2AddV2 conv_lstm2d_3/BiasAdd_1:output:0$conv_lstm2d_3/convolution_6:output:0*
T0*/
_output_shapes
:���������Z
conv_lstm2d_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��L>Z
conv_lstm2d_3/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?�
conv_lstm2d_3/Mul_1Mulconv_lstm2d_3/add_2:z:0conv_lstm2d_3/Const_2:output:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/Add_3AddV2conv_lstm2d_3/Mul_1:z:0conv_lstm2d_3/Const_3:output:0*
T0*/
_output_shapes
:���������l
'conv_lstm2d_3/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%conv_lstm2d_3/clip_by_value_1/MinimumMinimumconv_lstm2d_3/Add_3:z:00conv_lstm2d_3/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:���������d
conv_lstm2d_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv_lstm2d_3/clip_by_value_1Maximum)conv_lstm2d_3/clip_by_value_1/Minimum:z:0(conv_lstm2d_3/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/mul_2Mul!conv_lstm2d_3/clip_by_value_1:z:0"conv_lstm2d_3/convolution:output:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/add_4AddV2 conv_lstm2d_3/BiasAdd_2:output:0$conv_lstm2d_3/convolution_7:output:0*
T0*/
_output_shapes
:���������m
conv_lstm2d_3/ReluReluconv_lstm2d_3/add_4:z:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/mul_3Mulconv_lstm2d_3/clip_by_value:z:0 conv_lstm2d_3/Relu:activations:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/add_5AddV2conv_lstm2d_3/mul_2:z:0conv_lstm2d_3/mul_3:z:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/add_6AddV2 conv_lstm2d_3/BiasAdd_3:output:0$conv_lstm2d_3/convolution_8:output:0*
T0*/
_output_shapes
:���������Z
conv_lstm2d_3/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *��L>Z
conv_lstm2d_3/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?�
conv_lstm2d_3/Mul_4Mulconv_lstm2d_3/add_6:z:0conv_lstm2d_3/Const_4:output:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/Add_7AddV2conv_lstm2d_3/Mul_4:z:0conv_lstm2d_3/Const_5:output:0*
T0*/
_output_shapes
:���������l
'conv_lstm2d_3/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%conv_lstm2d_3/clip_by_value_2/MinimumMinimumconv_lstm2d_3/Add_7:z:00conv_lstm2d_3/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:���������d
conv_lstm2d_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv_lstm2d_3/clip_by_value_2Maximum)conv_lstm2d_3/clip_by_value_2/Minimum:z:0(conv_lstm2d_3/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:���������o
conv_lstm2d_3/Relu_1Reluconv_lstm2d_3/add_5:z:0*
T0*/
_output_shapes
:����������
conv_lstm2d_3/mul_5Mul!conv_lstm2d_3/clip_by_value_2:z:0"conv_lstm2d_3/Relu_1:activations:0*
T0*/
_output_shapes
:����������
+conv_lstm2d_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         l
*conv_lstm2d_3/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
conv_lstm2d_3/TensorArrayV2_1TensorListReserve4conv_lstm2d_3/TensorArrayV2_1/element_shape:output:03conv_lstm2d_3/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���T
conv_lstm2d_3/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&conv_lstm2d_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������b
 conv_lstm2d_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
conv_lstm2d_3/whileWhile)conv_lstm2d_3/while/loop_counter:output:0/conv_lstm2d_3/while/maximum_iterations:output:0conv_lstm2d_3/time:output:0&conv_lstm2d_3/TensorArrayV2_1:handle:0"conv_lstm2d_3/convolution:output:0"conv_lstm2d_3/convolution:output:0$conv_lstm2d_3/strided_slice:output:0Econv_lstm2d_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0+conv_lstm2d_3_split_readvariableop_resource-conv_lstm2d_3_split_1_readvariableop_resource-conv_lstm2d_3_split_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( **
body"R 
conv_lstm2d_3_while_body_70508**
cond"R 
conv_lstm2d_3_while_cond_70507*[
output_shapesJ
H: : : : :���������:���������: : : : : *
parallel_iterations �
>conv_lstm2d_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
0conv_lstm2d_3/TensorArrayV2Stack/TensorListStackTensorListStackconv_lstm2d_3/while:output:3Gconv_lstm2d_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:���������*
element_dtype0*
num_elementsv
#conv_lstm2d_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������o
%conv_lstm2d_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%conv_lstm2d_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv_lstm2d_3/strided_slice_2StridedSlice9conv_lstm2d_3/TensorArrayV2Stack/TensorListStack:tensor:0,conv_lstm2d_3/strided_slice_2/stack:output:0.conv_lstm2d_3/strided_slice_2/stack_1:output:0.conv_lstm2d_3/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:���������*
shrink_axis_mask{
conv_lstm2d_3/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                �
conv_lstm2d_3/transpose_1	Transpose9conv_lstm2d_3/TensorArrayV2Stack/TensorListStack:tensor:0'conv_lstm2d_3/transpose_1/perm:output:0*
T0*3
_output_shapes!
:����������
dropout_12/IdentityIdentity&conv_lstm2d_3/strided_slice_2:output:0*
T0*/
_output_shapes
:����������
max_pooling2d_3/MaxPoolMaxPooldropout_12/Identity:output:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
{
dropout_13/IdentityIdentity max_pooling2d_3/MaxPool:output:0*
T0*/
_output_shapes
:���������`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����'   �
flatten_3/ReshapeReshapedropout_13/Identity:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:���������'�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	'�*
dtype0�
dense_9/MatMulMatMulflatten_3/Reshape:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:����������n
dropout_14/IdentityIdentitydense_9/Relu:activations:0*
T0*(
_output_shapes
:�����������
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes
:	�2*
dtype0�
dense_10/MatMulMatMuldropout_14/Identity:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2b
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:���������2n
dropout_15/IdentityIdentitydense_10/Relu:activations:0*
T0*'
_output_shapes
:���������2�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
dense_11/MatMulMatMuldropout_15/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_11/SigmoidSigmoiddense_11/BiasAdd:output:0*
T0*'
_output_shapes
:���������c
IdentityIdentitydense_11/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^conv_lstm2d_3/split/ReadVariableOp%^conv_lstm2d_3/split_1/ReadVariableOp%^conv_lstm2d_3/split_2/ReadVariableOp^conv_lstm2d_3/while ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:����������: : : : : : : : : 2H
"conv_lstm2d_3/split/ReadVariableOp"conv_lstm2d_3/split/ReadVariableOp2L
$conv_lstm2d_3/split_1/ReadVariableOp$conv_lstm2d_3/split_1/ReadVariableOp2L
$conv_lstm2d_3/split_2/ReadVariableOp$conv_lstm2d_3/split_2/ReadVariableOp2*
conv_lstm2d_3/whileconv_lstm2d_3/while2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
0__inference_conv_lstm_cell_3_layer_call_fn_72137

inputs
states_0
states_1"
unknown:�4#
	unknown_0:4
	unknown_1:4
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:���������:���������:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_conv_lstm_cell_3_layer_call_and_return_conditional_losses_69266w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:���������y

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:���������:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:YU
/
_output_shapes
:���������
"
_user_specified_name
states/0:YU
/
_output_shapes
:���������
"
_user_specified_name
states/1"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
P
reshape_3_input=
!serving_default_reshape_3_input:0����������<
dense_110
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-1
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer-9
layer_with_weights-3
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses
!cell
"
state_spec"
_tf_keras_rnn_layer
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)_random_generator"
_tf_keras_layer
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6_random_generator"
_tf_keras_layer
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses"
_tf_keras_layer
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

Ckernel
Dbias"
_tf_keras_layer
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses
K_random_generator"
_tf_keras_layer
�
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses

Rkernel
Sbias"
_tf_keras_layer
�
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses
Z_random_generator"
_tf_keras_layer
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses

akernel
bbias"
_tf_keras_layer
_
c0
d1
e2
C3
D4
R5
S6
a7
b8"
trackable_list_wrapper
_
c0
d1
e2
C3
D4
R5
S6
a7
b8"
trackable_list_wrapper
 "
trackable_list_wrapper
�
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
ktrace_0
ltrace_1
mtrace_2
ntrace_32�
,__inference_sequential_3_layer_call_fn_69775
,__inference_sequential_3_layer_call_fn_70377
,__inference_sequential_3_layer_call_fn_70400
,__inference_sequential_3_layer_call_fn_70257�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zktrace_0zltrace_1zmtrace_2zntrace_3
�
otrace_0
ptrace_1
qtrace_2
rtrace_32�
G__inference_sequential_3_layer_call_and_return_conditional_losses_70663
G__inference_sequential_3_layer_call_and_return_conditional_losses_70954
G__inference_sequential_3_layer_call_and_return_conditional_losses_70290
G__inference_sequential_3_layer_call_and_return_conditional_losses_70323�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zotrace_0zptrace_1zqtrace_2zrtrace_3
�B�
 __inference__wrapped_model_68972reshape_3_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
siter

tbeta_1

ubeta_2
	vdecay
wlearning_rateCm�Dm�Rm�Sm�am�bm�cm�dm�em�Cv�Dv�Rv�Sv�av�bv�cv�dv�ev�"
	optimizer
,
xserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
~trace_02�
)__inference_reshape_3_layer_call_fn_70959�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z~trace_0
�
trace_02�
D__inference_reshape_3_layer_call_and_return_conditional_losses_70974�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
5
c0
d1
e2"
trackable_list_wrapper
5
c0
d1
e2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�states
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
-__inference_conv_lstm2d_3_layer_call_fn_70985
-__inference_conv_lstm2d_3_layer_call_fn_70996
-__inference_conv_lstm2d_3_layer_call_fn_71007
-__inference_conv_lstm2d_3_layer_call_fn_71018�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_71242
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_71466
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_71690
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_71914�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator

ckernel
drecurrent_kernel
ebias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_12_layer_call_fn_71919
*__inference_dropout_12_layer_call_fn_71924�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_12_layer_call_and_return_conditional_losses_71929
E__inference_dropout_12_layer_call_and_return_conditional_losses_71941�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_max_pooling2d_3_layer_call_fn_71946�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_71951�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_13_layer_call_fn_71956
*__inference_dropout_13_layer_call_fn_71961�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_13_layer_call_and_return_conditional_losses_71966
E__inference_dropout_13_layer_call_and_return_conditional_losses_71978�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_flatten_3_layer_call_fn_71983�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_flatten_3_layer_call_and_return_conditional_losses_71989�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_dense_9_layer_call_fn_71998�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_dense_9_layer_call_and_return_conditional_losses_72009�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:	'�2dense_9/kernel
:�2dense_9/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_14_layer_call_fn_72014
*__inference_dropout_14_layer_call_fn_72019�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_14_layer_call_and_return_conditional_losses_72024
E__inference_dropout_14_layer_call_and_return_conditional_losses_72036�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_10_layer_call_fn_72045�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_10_layer_call_and_return_conditional_losses_72056�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 	�22dense_10/kernel
:22dense_10/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_15_layer_call_fn_72061
*__inference_dropout_15_layer_call_fn_72066�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_15_layer_call_and_return_conditional_losses_72071
E__inference_dropout_15_layer_call_and_return_conditional_losses_72083�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
a0
b1"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_11_layer_call_fn_72092�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_11_layer_call_and_return_conditional_losses_72103�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:22dense_11/kernel
:2dense_11/bias
/:-�42conv_lstm2d_3/kernel
8:642conv_lstm2d_3/recurrent_kernel
 :42conv_lstm2d_3/bias
 "
trackable_list_wrapper
n
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
10"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_sequential_3_layer_call_fn_69775reshape_3_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_sequential_3_layer_call_fn_70377inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_sequential_3_layer_call_fn_70400inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_sequential_3_layer_call_fn_70257reshape_3_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_3_layer_call_and_return_conditional_losses_70663inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_3_layer_call_and_return_conditional_losses_70954inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_3_layer_call_and_return_conditional_losses_70290reshape_3_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_3_layer_call_and_return_conditional_losses_70323reshape_3_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
#__inference_signature_wrapper_70354reshape_3_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_reshape_3_layer_call_fn_70959inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_reshape_3_layer_call_and_return_conditional_losses_70974inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
!0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_conv_lstm2d_3_layer_call_fn_70985inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_conv_lstm2d_3_layer_call_fn_70996inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_conv_lstm2d_3_layer_call_fn_71007inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_conv_lstm2d_3_layer_call_fn_71018inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_71242inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_71466inputs/0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_71690inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_71914inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
5
c0
d1
e2"
trackable_list_wrapper
5
c0
d1
e2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
0__inference_conv_lstm_cell_3_layer_call_fn_72120
0__inference_conv_lstm_cell_3_layer_call_fn_72137�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
K__inference_conv_lstm_cell_3_layer_call_and_return_conditional_losses_72212
K__inference_conv_lstm_cell_3_layer_call_and_return_conditional_losses_72287�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dropout_12_layer_call_fn_71919inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_dropout_12_layer_call_fn_71924inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_12_layer_call_and_return_conditional_losses_71929inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_12_layer_call_and_return_conditional_losses_71941inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_max_pooling2d_3_layer_call_fn_71946inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_71951inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dropout_13_layer_call_fn_71956inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_dropout_13_layer_call_fn_71961inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_13_layer_call_and_return_conditional_losses_71966inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_13_layer_call_and_return_conditional_losses_71978inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_flatten_3_layer_call_fn_71983inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_flatten_3_layer_call_and_return_conditional_losses_71989inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_dense_9_layer_call_fn_71998inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dense_9_layer_call_and_return_conditional_losses_72009inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dropout_14_layer_call_fn_72014inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_dropout_14_layer_call_fn_72019inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_14_layer_call_and_return_conditional_losses_72024inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_14_layer_call_and_return_conditional_losses_72036inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_10_layer_call_fn_72045inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_10_layer_call_and_return_conditional_losses_72056inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dropout_15_layer_call_fn_72061inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_dropout_15_layer_call_fn_72066inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_15_layer_call_and_return_conditional_losses_72071inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_15_layer_call_and_return_conditional_losses_72083inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_11_layer_call_fn_72092inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_11_layer_call_and_return_conditional_losses_72103inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_conv_lstm_cell_3_layer_call_fn_72120inputsstates/0states/1"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
0__inference_conv_lstm_cell_3_layer_call_fn_72137inputsstates/0states/1"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_conv_lstm_cell_3_layer_call_and_return_conditional_losses_72212inputsstates/0states/1"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_conv_lstm_cell_3_layer_call_and_return_conditional_losses_72287inputsstates/0states/1"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
&:$	'�2Adam/dense_9/kernel/m
 :�2Adam/dense_9/bias/m
':%	�22Adam/dense_10/kernel/m
 :22Adam/dense_10/bias/m
&:$22Adam/dense_11/kernel/m
 :2Adam/dense_11/bias/m
4:2�42Adam/conv_lstm2d_3/kernel/m
=:;42%Adam/conv_lstm2d_3/recurrent_kernel/m
%:#42Adam/conv_lstm2d_3/bias/m
&:$	'�2Adam/dense_9/kernel/v
 :�2Adam/dense_9/bias/v
':%	�22Adam/dense_10/kernel/v
 :22Adam/dense_10/bias/v
&:$22Adam/dense_11/kernel/v
 :2Adam/dense_11/bias/v
4:2�42Adam/conv_lstm2d_3/kernel/v
=:;42%Adam/conv_lstm2d_3/recurrent_kernel/v
%:#42Adam/conv_lstm2d_3/bias/v�
 __inference__wrapped_model_68972	cdeCDRSab=�:
3�0
.�+
reshape_3_input����������
� "3�0
.
dense_11"�
dense_11����������
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_71242�cdeX�U
N�K
=�:
8�5
inputs/0'�������������������

 
p 

 
� "-�*
#� 
0���������
� �
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_71466�cdeX�U
N�K
=�:
8�5
inputs/0'�������������������

 
p

 
� "-�*
#� 
0���������
� �
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_71690~cdeH�E
>�;
-�*
inputs����������

 
p 

 
� "-�*
#� 
0���������
� �
H__inference_conv_lstm2d_3_layer_call_and_return_conditional_losses_71914~cdeH�E
>�;
-�*
inputs����������

 
p

 
� "-�*
#� 
0���������
� �
-__inference_conv_lstm2d_3_layer_call_fn_70985�cdeX�U
N�K
=�:
8�5
inputs/0'�������������������

 
p 

 
� " �����������
-__inference_conv_lstm2d_3_layer_call_fn_70996�cdeX�U
N�K
=�:
8�5
inputs/0'�������������������

 
p

 
� " �����������
-__inference_conv_lstm2d_3_layer_call_fn_71007qcdeH�E
>�;
-�*
inputs����������

 
p 

 
� " �����������
-__inference_conv_lstm2d_3_layer_call_fn_71018qcdeH�E
>�;
-�*
inputs����������

 
p

 
� " �����������
K__inference_conv_lstm_cell_3_layer_call_and_return_conditional_losses_72212�cde���
���
)�&
inputs����������
[�X
*�'
states/0���������
*�'
states/1���������
p 
� "���
��~
%�"
0/0���������
U�R
'�$
0/1/0���������
'�$
0/1/1���������
� �
K__inference_conv_lstm_cell_3_layer_call_and_return_conditional_losses_72287�cde���
���
)�&
inputs����������
[�X
*�'
states/0���������
*�'
states/1���������
p
� "���
��~
%�"
0/0���������
U�R
'�$
0/1/0���������
'�$
0/1/1���������
� �
0__inference_conv_lstm_cell_3_layer_call_fn_72120�cde���
���
)�&
inputs����������
[�X
*�'
states/0���������
*�'
states/1���������
p 
� "{�x
#� 
0���������
Q�N
%�"
1/0���������
%�"
1/1����������
0__inference_conv_lstm_cell_3_layer_call_fn_72137�cde���
���
)�&
inputs����������
[�X
*�'
states/0���������
*�'
states/1���������
p
� "{�x
#� 
0���������
Q�N
%�"
1/0���������
%�"
1/1����������
C__inference_dense_10_layer_call_and_return_conditional_losses_72056]RS0�-
&�#
!�
inputs����������
� "%�"
�
0���������2
� |
(__inference_dense_10_layer_call_fn_72045PRS0�-
&�#
!�
inputs����������
� "����������2�
C__inference_dense_11_layer_call_and_return_conditional_losses_72103\ab/�,
%�"
 �
inputs���������2
� "%�"
�
0���������
� {
(__inference_dense_11_layer_call_fn_72092Oab/�,
%�"
 �
inputs���������2
� "�����������
B__inference_dense_9_layer_call_and_return_conditional_losses_72009]CD/�,
%�"
 �
inputs���������'
� "&�#
�
0����������
� {
'__inference_dense_9_layer_call_fn_71998PCD/�,
%�"
 �
inputs���������'
� "������������
E__inference_dropout_12_layer_call_and_return_conditional_losses_71929l;�8
1�.
(�%
inputs���������
p 
� "-�*
#� 
0���������
� �
E__inference_dropout_12_layer_call_and_return_conditional_losses_71941l;�8
1�.
(�%
inputs���������
p
� "-�*
#� 
0���������
� �
*__inference_dropout_12_layer_call_fn_71919_;�8
1�.
(�%
inputs���������
p 
� " �����������
*__inference_dropout_12_layer_call_fn_71924_;�8
1�.
(�%
inputs���������
p
� " �����������
E__inference_dropout_13_layer_call_and_return_conditional_losses_71966l;�8
1�.
(�%
inputs���������
p 
� "-�*
#� 
0���������
� �
E__inference_dropout_13_layer_call_and_return_conditional_losses_71978l;�8
1�.
(�%
inputs���������
p
� "-�*
#� 
0���������
� �
*__inference_dropout_13_layer_call_fn_71956_;�8
1�.
(�%
inputs���������
p 
� " �����������
*__inference_dropout_13_layer_call_fn_71961_;�8
1�.
(�%
inputs���������
p
� " �����������
E__inference_dropout_14_layer_call_and_return_conditional_losses_72024^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
E__inference_dropout_14_layer_call_and_return_conditional_losses_72036^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� 
*__inference_dropout_14_layer_call_fn_72014Q4�1
*�'
!�
inputs����������
p 
� "�����������
*__inference_dropout_14_layer_call_fn_72019Q4�1
*�'
!�
inputs����������
p
� "������������
E__inference_dropout_15_layer_call_and_return_conditional_losses_72071\3�0
)�&
 �
inputs���������2
p 
� "%�"
�
0���������2
� �
E__inference_dropout_15_layer_call_and_return_conditional_losses_72083\3�0
)�&
 �
inputs���������2
p
� "%�"
�
0���������2
� }
*__inference_dropout_15_layer_call_fn_72061O3�0
)�&
 �
inputs���������2
p 
� "����������2}
*__inference_dropout_15_layer_call_fn_72066O3�0
)�&
 �
inputs���������2
p
� "����������2�
D__inference_flatten_3_layer_call_and_return_conditional_losses_71989`7�4
-�*
(�%
inputs���������
� "%�"
�
0���������'
� �
)__inference_flatten_3_layer_call_fn_71983S7�4
-�*
(�%
inputs���������
� "����������'�
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_71951�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
/__inference_max_pooling2d_3_layer_call_fn_71946�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
D__inference_reshape_3_layer_call_and_return_conditional_losses_70974j4�1
*�'
%�"
inputs����������
� "2�/
(�%
0����������
� �
)__inference_reshape_3_layer_call_fn_70959]4�1
*�'
%�"
inputs����������
� "%�"�����������
G__inference_sequential_3_layer_call_and_return_conditional_losses_70290y	cdeCDRSabE�B
;�8
.�+
reshape_3_input����������
p 

 
� "%�"
�
0���������
� �
G__inference_sequential_3_layer_call_and_return_conditional_losses_70323y	cdeCDRSabE�B
;�8
.�+
reshape_3_input����������
p

 
� "%�"
�
0���������
� �
G__inference_sequential_3_layer_call_and_return_conditional_losses_70663p	cdeCDRSab<�9
2�/
%�"
inputs����������
p 

 
� "%�"
�
0���������
� �
G__inference_sequential_3_layer_call_and_return_conditional_losses_70954p	cdeCDRSab<�9
2�/
%�"
inputs����������
p

 
� "%�"
�
0���������
� �
,__inference_sequential_3_layer_call_fn_69775l	cdeCDRSabE�B
;�8
.�+
reshape_3_input����������
p 

 
� "�����������
,__inference_sequential_3_layer_call_fn_70257l	cdeCDRSabE�B
;�8
.�+
reshape_3_input����������
p

 
� "�����������
,__inference_sequential_3_layer_call_fn_70377c	cdeCDRSab<�9
2�/
%�"
inputs����������
p 

 
� "�����������
,__inference_sequential_3_layer_call_fn_70400c	cdeCDRSab<�9
2�/
%�"
inputs����������
p

 
� "�����������
#__inference_signature_wrapper_70354�	cdeCDRSabP�M
� 
F�C
A
reshape_3_input.�+
reshape_3_input����������"3�0
.
dense_11"�
dense_11���������