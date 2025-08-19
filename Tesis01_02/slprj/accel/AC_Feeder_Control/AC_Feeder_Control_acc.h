#ifndef RTW_HEADER_AC_Feeder_Control_acc_h_
#define RTW_HEADER_AC_Feeder_Control_acc_h_
#ifndef AC_Feeder_Control_acc_COMMON_INCLUDES_
#define AC_Feeder_Control_acc_COMMON_INCLUDES_
#include <stdlib.h>
#include <stddef.h>
#define S_FUNCTION_NAME simulink_only_sfcn
#define S_FUNCTION_LEVEL 2
#ifndef RTW_GENERATED_S_FUNCTION
#define RTW_GENERATED_S_FUNCTION
#endif
#include "sl_AsyncioQueue/AsyncioQueueCAPI.h"
#include "rtwtypes.h"
#include "sl_fileio_rtw.h"
#include "simtarget/slSimTgtSlFileioRTW.h"
#include "simstruc.h"
#include "fixedpoint.h"
#endif
#include "AC_Feeder_Control_acc_types.h"
#include <float.h>
#include "mwmathutil.h"
#include "rt_defines.h"
#include "rt_nonfinite.h"
#include "simstruc_types.h"
typedef struct { real_T B_12_11_0 ; real_T B_12_15_0 [ 3 ] ; real_T B_12_20_0
; real_T B_12_22_0 ; real_T B_12_23_0 ; real_T B_12_25_0 ; real_T B_12_27_0 ;
real_T B_12_29_0 ; real_T B_12_31_0 ; real_T B_12_33_0 ; real_T B_12_36_0 ;
real_T B_12_38_0 ; real_T B_12_40_0 ; real_T B_12_43_0 ; real_T B_12_45_0 ;
real_T B_12_46_0 [ 15 ] ; real_T B_12_46_1 [ 6 ] ; real_T B_12_63_0 [ 2 ] ;
real_T B_12_65_0 ; real_T B_12_67_0 ; real_T B_12_69_0 ; real_T B_12_70_0 ;
real_T B_12_71_0 ; real_T B_12_73_0 ; real_T B_12_79_0 ; real_T B_12_109_0 [
3 ] ; real_T B_12_117_0 ; real_T B_12_154_0 ; real_T B_12_157_0 ; real_T
B_12_160_0 ; real_T B_12_162_0 ; real_T B_12_166_0 ; real_T B_12_168_0 ;
real_T B_12_170_0 ; real_T B_12_174_0 ; real_T B_12_176_0 ; real_T B_12_178_0
; real_T B_12_189_0 ; real_T B_12_192_0 ; real_T B_12_195_0 ; real_T
B_12_197_0 ; real_T B_12_201_0 ; real_T B_12_203_0 ; real_T B_12_205_0 ;
real_T B_12_209_0 ; real_T B_12_211_0 ; real_T B_12_213_0 ; real_T B_12_0_0 ;
real_T B_12_4_0 ; real_T B_12_5_0 [ 3 ] ; real_T B_12_7_0 ; real_T B_12_8_0 ;
real_T B_12_9_0 [ 3 ] ; real_T B_12_10_0 ; real_T B_12_11_0_m ; real_T
B_12_12_0 ; real_T B_12_13_0 ; real_T B_12_14_0 ; real_T B_12_15_0_c ; real_T
B_12_16_0 ; real_T B_12_19_0 ; real_T B_12_20_0_k ; real_T B_12_26_0 ; real_T
B_12_27_0_c ; real_T B_12_28_0 ; real_T B_12_29_0_b ; real_T B_12_30_0 ;
real_T B_12_31_0_p ; real_T B_12_32_0 ; real_T B_12_33_0_c ; real_T B_12_34_0
; real_T B_12_35_0 ; real_T B_12_36_0_f [ 6 ] ; real_T B_5_0_0 ; real_T
B_4_1_0 ; real_T B_4_3_0 ; real_T B_4_4_0 ; real_T B_4_7_0 ; real_T B_4_8_0 ;
real_T B_4_10_0 ; real_T B_2_0_0 ; real_T B_2_3_0 ; real_T B_2_5_0 ; real_T
B_2_7_0 ; real_T B_2_8_0 ; real_T B_2_11_0 ; real_T B_2_13_0 ; real_T
B_2_15_0 ; real_T B_2_19_0 ; real_T B_2_21_0 ; real_T B_2_22_0 ; real_T
B_12_50_0 [ 3 ] ; real32_T B_12_17_0 ; real32_T B_12_18_0 ; boolean_T
B_12_10_0_g ; boolean_T B_12_21_0 ; boolean_T B_12_21_0_g ; boolean_T
B_12_24_0 ; boolean_T B_12_25_0_m ; char_T pad_B_12_25_0_m [ 3 ] ; }
B_AC_Feeder_Control_T ; typedef struct { real_T UnitDelay_DSTATE ; real_T
DiscreteTimeIntegrator1_DSTATE ; real_T UnitDelay_DSTATE_e ; real_T
DiscreteTimeIntegrator1_DSTATE_l ; real_T StateSpace_DSTATE [ 6 ] ; real_T
Sum3_DWORK1 ; real_T Sum1_DWORK1 ; real_T Memory_PreviousInput ; real_T
Memory_PreviousInput_h ; real_T Memory_PreviousInput_hn ; struct { real_T
modelTStart ; } TransportDelay_RWORK ; struct { real_T modelTStart ; }
TransportDelay_RWORK_e ; struct { real_T modelTStart ; }
TransportDelay_RWORK_i ; struct { void * AS ; void * BS ; void * CS ; void *
DS ; void * DX_COL ; void * BD_COL ; void * TMP1 ; void * TMP2 ; void * XTMP
; void * SWITCH_STATUS ; void * SWITCH_STATUS_INIT ; void * SW_CHG ; void *
G_STATE ; void * USWLAST ; void * XKM12 ; void * XKP12 ; void * XLAST ; void
* ULAST ; void * IDX_SW_CHG ; void * Y_SWITCH ; void * SWITCH_TYPES ; void *
IDX_OUT_SW ; void * SWITCH_TOPO_SAVED_IDX ; void * SWITCH_MAP ; }
StateSpace_PWORK ; void * Scope_PWORK [ 4 ] ; void * Scope1_PWORK [ 4 ] ;
void * Scope_PWORK_c [ 4 ] ; void * Scope_PWORK_d [ 4 ] ; void *
Scope1_PWORK_l [ 3 ] ; struct { void * AQHandles ; }
TAQSigLogging_InsertedFor_Clk_at_outport_0_PWORK ; struct { void * FilePtr ;
} ToFile_PWORK ; struct { void * TUbufferPtrs [ 2 ] ; } TransportDelay_PWORK
; struct { void * TUbufferPtrs [ 2 ] ; } TransportDelay_PWORK_i ; struct {
void * TUbufferPtrs [ 2 ] ; } TransportDelay_PWORK_i3 ; int32_T
TmpAtomicSubsysAtSwitch3Inport1_sysIdxToRun ; int32_T
TmpAtomicSubsysAtSwitch3Inport1_sysIdxToRun_a ; int32_T
TmpAtomicSubsysAtSwitch3Inport1_sysIdxToRun_h ; int32_T
TmpAtomicSubsysAtSwitch3Inport1_sysIdxToRun_g ; int32_T
TmpAtomicSubsysAtSwitch3Inport1_sysIdxToRun_ad ; int32_T
TmpAtomicSubsysAtSwitch3Inport1_sysIdxToRun_as ; int32_T Sampling_sysIdxToRun
; int32_T TrueRMS_sysIdxToRun ; int32_T
TmpAtomicSubsysAtSwitchInport1_sysIdxToRun ; int32_T RMS_sysIdxToRun ;
int32_T TmpAtomicSubsysAtSwitchInport1_sysIdxToRun_g ; int32_T
TmpAtomicSubsysAtSwitchInport1_sysIdxToRun_gf ; int_T StateSpace_IWORK [ 11 ]
; struct { int_T Count ; int_T Decimation ; } ToFile_IWORK ; struct { int_T
Tail ; int_T Head ; int_T Last ; int_T CircularBufSize ; int_T MaxNewBufSize
; } TransportDelay_IWORK ; struct { int_T Tail ; int_T Head ; int_T Last ;
int_T CircularBufSize ; int_T MaxNewBufSize ; } TransportDelay_IWORK_j ;
struct { int_T Tail ; int_T Head ; int_T Last ; int_T CircularBufSize ; int_T
MaxNewBufSize ; } TransportDelay_IWORK_a ; int8_T
DiscreteTimeIntegrator1_PrevResetState ; int8_T
DiscreteTimeIntegrator1_PrevResetState_m ; int8_T Sampling_SubsysRanBC ;
int8_T TrueRMS_SubsysRanBC ; int8_T RMS_SubsysRanBC ; boolean_T TrueRMS_MODE
; boolean_T RMS_MODE ; char_T pad_RMS_MODE [ 1 ] ; } DW_AC_Feeder_Control_T ;
typedef struct { real_T Integrator1_CSTATE ; real_T Integrator_CSTATE ;
real_T integrator_CSTATE ; real_T integrator_CSTATE_l ; real_T
integrator_CSTATE_k ; } X_AC_Feeder_Control_T ; typedef struct { real_T
Integrator1_CSTATE ; real_T Integrator_CSTATE ; real_T integrator_CSTATE ;
real_T integrator_CSTATE_l ; real_T integrator_CSTATE_k ; }
XDot_AC_Feeder_Control_T ; typedef struct { boolean_T Integrator1_CSTATE ;
boolean_T Integrator_CSTATE ; boolean_T integrator_CSTATE ; boolean_T
integrator_CSTATE_l ; boolean_T integrator_CSTATE_k ; }
XDis_AC_Feeder_Control_T ; typedef struct { ZCSigState Sampling_Trig_ZCE ; }
PrevZCX_AC_Feeder_Control_T ; struct P_AC_Feeder_Control_T_ { real_T P_0 ;
real_T P_1 ; real_T P_2 ; real_T P_3 ; real_T P_4 ; real_T P_5 ; real_T P_6 ;
real_T P_7 ; real_T P_8 ; real_T P_9 ; real_T P_10 ; real_T P_11 ; real_T
P_12 ; real_T P_13 ; real_T P_14 ; real_T P_15 ; real_T P_16 ; real_T P_17 ;
real_T P_18 ; real_T P_19 ; real_T P_20 ; real_T P_21 ; real_T P_22 ; real_T
P_23 ; real_T P_24 ; real_T P_25 ; real_T P_26 ; real_T P_27 ; real_T P_28 ;
real_T P_29 ; real_T P_30 ; real_T P_31 ; real_T P_32 ; real_T P_33 ; real_T
P_34 ; real_T P_35 ; real_T P_36 ; real_T P_37 ; real_T P_38 ; real_T P_39 ;
real_T P_40 ; real_T P_41 ; real_T P_42 ; real_T P_43 ; real_T P_44 ; real_T
P_45 ; real_T P_46 [ 2 ] ; real_T P_47 [ 36 ] ; real_T P_48 [ 2 ] ; real_T
P_49 [ 72 ] ; real_T P_50 [ 2 ] ; real_T P_51 [ 90 ] ; real_T P_52 [ 2 ] ;
real_T P_53 [ 180 ] ; real_T P_54 [ 2 ] ; real_T P_55 [ 6 ] ; real_T P_56 ;
real_T P_57 ; real_T P_58 ; real_T P_59 ; real_T P_60 ; real_T P_61 ; real_T
P_62 ; real_T P_63 ; real_T P_64 ; real_T P_65 ; real_T P_66 ; real_T P_67 ;
real_T P_68 ; real_T P_69 ; real_T P_70 ; real_T P_71 ; real_T P_72 ; real_T
P_73 ; real_T P_74 ; real_T P_75 ; real_T P_76 ; real_T P_77 ; real_T P_78 ;
real_T P_79 ; real_T P_80 ; real_T P_81 ; real_T P_82 [ 4 ] ; real_T P_83 [ 4
] ; real_T P_84 ; real_T P_85 [ 5 ] ; real_T P_86 [ 5 ] ; real_T P_87 ;
real_T P_88 ; real_T P_89 [ 5 ] ; real_T P_90 [ 5 ] ; real_T P_91 ; real_T
P_92 ; real_T P_93 [ 5 ] ; real_T P_94 [ 5 ] ; real_T P_95 ; real_T P_96 ;
real_T P_97 ; real_T P_98 ; real_T P_99 ; real_T P_100 [ 4 ] ; real_T P_101 [
4 ] ; real_T P_102 ; real_T P_103 [ 5 ] ; real_T P_104 [ 5 ] ; real_T P_105 ;
real_T P_106 ; real_T P_107 [ 5 ] ; real_T P_108 [ 5 ] ; real_T P_109 ;
real_T P_110 ; real_T P_111 [ 5 ] ; real_T P_112 [ 5 ] ; real_T P_113 ;
real_T P_114 ; real_T P_115 ; real_T P_116 ; real_T P_117 ; real_T P_118 [ 3
] ; real_T P_119 [ 3 ] ; real_T P_120 ; real_T P_121 ; real_T P_122 ; real_T
P_123 ; real_T P_124 ; real_T P_125 ; real_T P_126 ; real_T P_127 ; real_T
P_128 ; real_T P_129 ; real_T P_130 ; real_T P_131 ; real_T P_132 ; real_T
P_133 ; real_T P_134 ; real_T P_135 ; real_T P_136 ; real_T P_137 ; real_T
P_138 ; real_T P_139 ; real_T P_140 ; real_T P_141 ; real_T P_142 ; real_T
P_143 [ 6 ] ; real32_T P_144 ; real32_T P_145 ; boolean_T P_146 ; char_T
pad_P_146 [ 7 ] ; } ; extern P_AC_Feeder_Control_T
AC_Feeder_Control_rtDefaultP ;
#endif
