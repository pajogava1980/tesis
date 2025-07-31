#include "AC_Feeder_Control_acc.h"
#include "mwmathutil.h"
#include "rt_look.h"
#include "rt_look1d.h"
#include "rtwtypes.h"
#include "AC_Feeder_Control_acc_private.h"
#include "multiword_types.h"
#include "simstruc_types.h"
#include <stdio.h>
#include "slexec_vm_simstruct_bridge.h"
#include "slexec_vm_zc_functions.h"
#include "slexec_vm_lookup_functions.h"
#include "slsv_diagnostic_codegen_c_api.h"
#include "simtarget/slSimTgtMdlrefSfcnBridge.h"
#include "simstruc.h"
#include "fixedpoint.h"
#define CodeFormat S-Function
#define AccDefine1 Accelerator_S-Function
#include "simtarget/slAccSfcnBridge.h"
#ifndef __RTW_UTFREE__  
extern void * utMalloc ( size_t ) ; extern void utFree ( void * ) ;
#endif
boolean_T AC_Feeder_Control_acc_rt_TDelayUpdateTailOrGrowBuf ( int_T *
bufSzPtr , int_T * tailPtr , int_T * headPtr , int_T * lastPtr , real_T
tMinusDelay , real_T * * uBufPtr , boolean_T isfixedbuf , boolean_T
istransportdelay , int_T * maxNewBufSzPtr ) { int_T testIdx ; int_T tail = *
tailPtr ; int_T bufSz = * bufSzPtr ; real_T * tBuf = * uBufPtr + bufSz ;
real_T * xBuf = ( NULL ) ; int_T numBuffer = 2 ; if ( istransportdelay ) {
numBuffer = 3 ; xBuf = * uBufPtr + 2 * bufSz ; } testIdx = ( tail < ( bufSz -
1 ) ) ? ( tail + 1 ) : 0 ; if ( ( tMinusDelay <= tBuf [ testIdx ] ) && !
isfixedbuf ) { int_T j ; real_T * tempT ; real_T * tempU ; real_T * tempX = (
NULL ) ; real_T * uBuf = * uBufPtr ; int_T newBufSz = bufSz + 1024 ; if (
newBufSz > * maxNewBufSzPtr ) { * maxNewBufSzPtr = newBufSz ; } tempU = (
real_T * ) utMalloc ( numBuffer * newBufSz * sizeof ( real_T ) ) ; if ( tempU
== ( NULL ) ) { return ( false ) ; } tempT = tempU + newBufSz ; if (
istransportdelay ) tempX = tempT + newBufSz ; for ( j = tail ; j < bufSz ; j
++ ) { tempT [ j - tail ] = tBuf [ j ] ; tempU [ j - tail ] = uBuf [ j ] ; if
( istransportdelay ) tempX [ j - tail ] = xBuf [ j ] ; } for ( j = 0 ; j <
tail ; j ++ ) { tempT [ j + bufSz - tail ] = tBuf [ j ] ; tempU [ j + bufSz -
tail ] = uBuf [ j ] ; if ( istransportdelay ) tempX [ j + bufSz - tail ] =
xBuf [ j ] ; } if ( * lastPtr > tail ) { * lastPtr -= tail ; } else { *
lastPtr += ( bufSz - tail ) ; } * tailPtr = 0 ; * headPtr = bufSz ; utFree (
uBuf ) ; * bufSzPtr = newBufSz ; * uBufPtr = tempU ; } else { * tailPtr =
testIdx ; } return ( true ) ; } real_T
AC_Feeder_Control_acc_rt_TDelayInterpolate ( real_T tMinusDelay , real_T
tStart , real_T * uBuf , int_T bufSz , int_T * lastIdx , int_T oldestIdx ,
int_T newIdx , real_T initOutput , boolean_T discrete , boolean_T
minorStepAndTAtLastMajorOutput ) { int_T i ; real_T yout , t1 , t2 , u1 , u2
; real_T * tBuf = uBuf + bufSz ; if ( ( newIdx == 0 ) && ( oldestIdx == 0 )
&& ( tMinusDelay > tStart ) ) return initOutput ; if ( tMinusDelay <= tStart
) return initOutput ; if ( ( tMinusDelay <= tBuf [ oldestIdx ] ) ) { if (
discrete ) { return ( uBuf [ oldestIdx ] ) ; } else { int_T tempIdx =
oldestIdx + 1 ; if ( oldestIdx == bufSz - 1 ) tempIdx = 0 ; t1 = tBuf [
oldestIdx ] ; t2 = tBuf [ tempIdx ] ; u1 = uBuf [ oldestIdx ] ; u2 = uBuf [
tempIdx ] ; if ( t2 == t1 ) { if ( tMinusDelay >= t2 ) { yout = u2 ; } else {
yout = u1 ; } } else { real_T f1 = ( t2 - tMinusDelay ) / ( t2 - t1 ) ;
real_T f2 = 1.0 - f1 ; yout = f1 * u1 + f2 * u2 ; } return yout ; } } if (
minorStepAndTAtLastMajorOutput ) { if ( newIdx != 0 ) { if ( * lastIdx ==
newIdx ) { ( * lastIdx ) -- ; } newIdx -- ; } else { if ( * lastIdx == newIdx
) { * lastIdx = bufSz - 1 ; } newIdx = bufSz - 1 ; } } i = * lastIdx ; if (
tBuf [ i ] < tMinusDelay ) { while ( tBuf [ i ] < tMinusDelay ) { if ( i ==
newIdx ) break ; i = ( i < ( bufSz - 1 ) ) ? ( i + 1 ) : 0 ; } } else { while
( tBuf [ i ] >= tMinusDelay ) { i = ( i > 0 ) ? i - 1 : ( bufSz - 1 ) ; } i =
( i < ( bufSz - 1 ) ) ? ( i + 1 ) : 0 ; } * lastIdx = i ; if ( discrete ) {
double tempEps = ( DBL_EPSILON ) * 128.0 ; double localEps = tempEps *
muDoubleScalarAbs ( tBuf [ i ] ) ; if ( tempEps > localEps ) { localEps =
tempEps ; } localEps = localEps / 2.0 ; if ( tMinusDelay >= ( tBuf [ i ] -
localEps ) ) { yout = uBuf [ i ] ; } else { if ( i == 0 ) { yout = uBuf [
bufSz - 1 ] ; } else { yout = uBuf [ i - 1 ] ; } } } else { if ( i == 0 ) {
t1 = tBuf [ bufSz - 1 ] ; u1 = uBuf [ bufSz - 1 ] ; } else { t1 = tBuf [ i -
1 ] ; u1 = uBuf [ i - 1 ] ; } t2 = tBuf [ i ] ; u2 = uBuf [ i ] ; if ( t2 ==
t1 ) { if ( tMinusDelay >= t2 ) { yout = u2 ; } else { yout = u1 ; } } else {
real_T f1 = ( t2 - tMinusDelay ) / ( t2 - t1 ) ; real_T f2 = 1.0 - f1 ; yout
= f1 * u1 + f2 * u2 ; } } return ( yout ) ; } void rt_ssGetBlockPath (
SimStruct * S , int_T sysIdx , int_T blkIdx , char_T * * path ) {
_ssGetBlockPath ( S , sysIdx , blkIdx , path ) ; } void rt_ssSet_slErrMsg (
void * S , void * diag ) { SimStruct * castedS = ( SimStruct * ) S ; if ( !
_ssIsErrorStatusAslErrMsg ( castedS ) ) { _ssSet_slErrMsg ( castedS , diag )
; } else { _ssDiscardDiagnostic ( castedS , diag ) ; } } void
rt_ssReportDiagnosticAsWarning ( void * S , void * diag ) {
_ssReportDiagnosticAsWarning ( ( SimStruct * ) S , diag ) ; } void
rt_ssReportDiagnosticAsInfo ( void * S , void * diag ) {
_ssReportDiagnosticAsInfo ( ( SimStruct * ) S , diag ) ; } static void
mdlOutputs ( SimStruct * S , int_T tid ) { real_T B_4_2_0 ; real_T B_2_1_0 ;
real_T B_2_9_0 ; B_AC_Feeder_Control_T * _rtB ; DW_AC_Feeder_Control_T *
_rtDW ; P_AC_Feeder_Control_T * _rtP ; PrevZCX_AC_Feeder_Control_T * _rtZCE ;
X_AC_Feeder_Control_T * _rtX ; real_T rtb_B_12_155_0 ; int32_T isHit ;
boolean_T rtb_B_2_4_0 ; _rtDW = ( ( DW_AC_Feeder_Control_T * ) ssGetRootDWork
( S ) ) ; _rtZCE = ( ( PrevZCX_AC_Feeder_Control_T * ) _ssGetPrevZCSigState (
S ) ) ; _rtX = ( ( X_AC_Feeder_Control_T * ) ssGetContStates ( S ) ) ; _rtP =
( ( P_AC_Feeder_Control_T * ) ssGetModelRtp ( S ) ) ; _rtB = ( (
B_AC_Feeder_Control_T * ) _ssGetModelBlockIO ( S ) ) ; isHit = ssIsSampleHit
( S , 1 , 0 ) ; if ( isHit != 0 ) { _rtB -> B_12_10_0_g = ( _rtDW ->
UnitDelay_DSTATE >= _rtB -> B_12_4_0 ) ; if ( _rtB -> B_12_10_0_g && ( _rtDW
-> DiscreteTimeIntegrator1_PrevResetState <= 0 ) ) { _rtDW ->
DiscreteTimeIntegrator1_DSTATE = _rtB -> B_12_8_0 ; } _rtB -> B_12_11_0 =
_rtDW -> DiscreteTimeIntegrator1_DSTATE ; _rtB -> B_12_15_0 [ 0 ] =
muDoubleScalarCos ( _rtB -> B_12_5_0 [ 0 ] * _rtB -> B_12_11_0 + _rtB ->
B_12_9_0 [ 0 ] ) * _rtB -> B_12_10_0 ; _rtB -> B_12_15_0 [ 1 ] =
muDoubleScalarCos ( _rtB -> B_12_5_0 [ 1 ] * _rtB -> B_12_11_0 + _rtB ->
B_12_9_0 [ 1 ] ) * _rtB -> B_12_10_0 ; _rtB -> B_12_15_0 [ 2 ] =
muDoubleScalarCos ( _rtB -> B_12_5_0 [ 2 ] * _rtB -> B_12_11_0 + _rtB ->
B_12_9_0 [ 2 ] ) * _rtB -> B_12_10_0 ; _rtB -> B_12_20_0 = _rtDW ->
UnitDelay_DSTATE_e ; _rtB -> B_12_21_0 = ( _rtB -> B_12_20_0 >= _rtB ->
B_12_13_0 ) ; if ( _rtB -> B_12_21_0 && ( _rtDW ->
DiscreteTimeIntegrator1_PrevResetState_m <= 0 ) ) { _rtDW ->
DiscreteTimeIntegrator1_DSTATE_l = _rtB -> B_12_14_0 ; } _rtB -> B_12_22_0 =
_rtDW -> DiscreteTimeIntegrator1_DSTATE_l ; _rtB -> B_12_23_0 =
muDoubleScalarCos ( _rtB -> B_12_22_0 ) ; _rtB -> B_12_25_0 = _rtP -> P_39 *
muDoubleScalarSin ( _rtB -> B_12_22_0 ) ; } if ( ssGetTaskTime ( S , 0 ) <
_rtP -> P_40 ) { _rtB -> B_12_27_0 = _rtP -> P_41 ; } else { _rtB ->
B_12_27_0 = _rtP -> P_42 ; } _rtB -> B_12_29_0 = _rtP -> P_43 * _rtB ->
B_12_27_0 / _rtB -> B_12_11_0_m ; _rtB -> B_12_31_0 = _rtB -> B_12_23_0 *
_rtB -> B_12_29_0 + _rtB -> B_12_25_0 * _rtB -> B_12_12_0 ; isHit =
ssIsSampleHit ( S , 1 , 0 ) ; if ( isHit != 0 ) { _rtB -> B_12_33_0 =
muDoubleScalarCos ( _rtB -> B_12_22_0 + _rtB -> B_12_20_0_k ) ; _rtB ->
B_12_36_0 = muDoubleScalarSin ( _rtB -> B_12_22_0 + _rtB -> B_12_20_0_k ) *
_rtP -> P_44 ; } _rtB -> B_12_38_0 = _rtB -> B_12_33_0 * _rtB -> B_12_29_0 +
_rtB -> B_12_36_0 * _rtB -> B_12_12_0 ; isHit = ssIsSampleHit ( S , 1 , 0 ) ;
if ( isHit != 0 ) { _rtB -> B_12_40_0 = muDoubleScalarCos ( _rtB -> B_12_22_0
+ _rtB -> B_12_19_0 ) ; _rtB -> B_12_43_0 = muDoubleScalarSin ( _rtB ->
B_12_22_0 + _rtB -> B_12_19_0 ) * _rtP -> P_45 ; } _rtB -> B_12_45_0 = _rtB
-> B_12_40_0 * _rtB -> B_12_29_0 + _rtB -> B_12_43_0 * _rtB -> B_12_12_0 ;
isHit = ssIsSampleHit ( S , 1 , 0 ) ; if ( isHit != 0 ) { ssCallAccelRunBlock
( S , 12 , 46 , SS_CALL_MDL_OUTPUTS ) ; _rtB -> B_12_50_0 [ 0 ] = _rtP ->
P_56 * _rtB -> B_12_46_0 [ 6 ] ; _rtB -> B_12_50_0 [ 1 ] = _rtP -> P_57 *
_rtB -> B_12_46_0 [ 7 ] ; _rtB -> B_12_50_0 [ 2 ] = _rtP -> P_58 * _rtB ->
B_12_46_0 [ 8 ] ; _rtB -> B_12_63_0 [ 0 ] = ( ( muDoubleScalarCos ( _rtB ->
B_12_15_0_c + _rtB -> B_12_22_0 ) * _rtP -> P_59 * _rtB -> B_12_50_0 [ 1 ] +
_rtP -> P_59 * muDoubleScalarCos ( _rtB -> B_12_22_0 ) * _rtB -> B_12_50_0 [
0 ] ) + muDoubleScalarCos ( _rtB -> B_12_16_0 + _rtB -> B_12_22_0 ) * _rtP ->
P_59 * _rtB -> B_12_50_0 [ 2 ] ) * _rtP -> P_61 ; _rtB -> B_12_63_0 [ 1 ] = (
( muDoubleScalarSin ( _rtB -> B_12_15_0_c + _rtB -> B_12_22_0 ) * _rtP ->
P_60 * _rtB -> B_12_50_0 [ 1 ] + _rtP -> P_60 * muDoubleScalarSin ( _rtB ->
B_12_22_0 ) * _rtB -> B_12_50_0 [ 0 ] ) + muDoubleScalarSin ( _rtB ->
B_12_16_0 + _rtB -> B_12_22_0 ) * _rtP -> P_60 * _rtB -> B_12_50_0 [ 2 ] ) *
_rtP -> P_61 ; ssCallAccelRunBlock ( S , 12 , 64 , SS_CALL_MDL_OUTPUTS ) ; }
_rtB -> B_12_65_0 = _rtX -> Integrator1_CSTATE ; _rtB -> B_12_67_0 = _rtX ->
Integrator_CSTATE ; _rtB -> B_12_69_0 = ( _rtB -> B_12_67_0 - _rtP -> P_63 *
_rtB -> B_12_65_0 ) * _rtP -> P_65 ; if ( _rtB -> B_12_69_0 > _rtP -> P_66 )
{ _rtB -> B_12_70_0 = _rtP -> P_66 ; } else if ( _rtB -> B_12_69_0 < _rtP ->
P_67 ) { _rtB -> B_12_70_0 = _rtP -> P_67 ; } else { _rtB -> B_12_70_0 = _rtB
-> B_12_69_0 ; } _rtB -> B_12_71_0 = _rtB -> B_12_18_0 + _rtB -> B_12_70_0 ;
isHit = ssIsSampleHit ( S , 1 , 0 ) ; if ( isHit != 0 ) { ssCallAccelRunBlock
( S , 12 , 72 , SS_CALL_MDL_OUTPUTS ) ; _rtB -> B_12_73_0 = _rtB -> B_12_17_0
- _rtB -> B_12_63_0 [ 1 ] ; ssCallAccelRunBlock ( S , 12 , 74 ,
SS_CALL_MDL_OUTPUTS ) ; } _rtB -> B_12_79_0 = ( ( _rtB -> B_12_70_0 - _rtB ->
B_12_69_0 ) * _rtP -> P_69 + _rtB -> B_12_73_0 ) + _rtP -> P_68 * _rtB ->
B_12_67_0 ; isHit = ssIsSampleHit ( S , 1 , 0 ) ; if ( isHit != 0 ) { _rtB ->
B_12_109_0 [ 0 ] = _rtP -> P_70 * _rtB -> B_12_46_0 [ 9 ] * _rtP -> P_73 ;
_rtB -> B_12_109_0 [ 1 ] = _rtP -> P_71 * _rtB -> B_12_46_0 [ 10 ] * _rtP ->
P_73 ; _rtB -> B_12_109_0 [ 2 ] = _rtP -> P_72 * _rtB -> B_12_46_0 [ 11 ] *
_rtP -> P_73 ; _rtB -> B_12_50_0 [ 0 ] = _rtP -> P_74 * _rtB -> B_12_46_0 [
12 ] * _rtP -> P_77 ; _rtB -> B_12_50_0 [ 1 ] = _rtP -> P_75 * _rtB ->
B_12_46_0 [ 13 ] * _rtP -> P_77 ; _rtB -> B_12_50_0 [ 2 ] = _rtP -> P_76 *
_rtB -> B_12_46_0 [ 14 ] * _rtP -> P_77 ; ssCallAccelRunBlock ( S , 12 , 114
, SS_CALL_MDL_OUTPUTS ) ; if ( ssIsModeUpdateTimeStep ( S ) ) { if ( _rtB ->
B_12_24_0 ) { if ( ! _rtDW -> TrueRMS_MODE ) { if ( ssGetTaskTime ( S , 1 )
!= ssGetTStart ( S ) ) { ssSetBlockStateForSolverChangedAtMajorStep ( S ) ; }
_rtDW -> TrueRMS_MODE = true ; } } else if ( _rtDW -> TrueRMS_MODE ) {
ssSetBlockStateForSolverChangedAtMajorStep ( S ) ; _rtDW -> TrueRMS_MODE =
false ; } } } if ( _rtDW -> TrueRMS_MODE ) { _rtB -> B_4_1_0 = _rtX ->
integrator_CSTATE ; { real_T * * uBuffer = ( real_T * * ) & _rtDW ->
TransportDelay_PWORK . TUbufferPtrs [ 0 ] ; real_T simTime = ssGetT ( S ) ;
real_T tMinusDelay = simTime - _rtP -> P_23 ; B_4_2_0 =
AC_Feeder_Control_acc_rt_TDelayInterpolate ( tMinusDelay , 0.0 , * uBuffer ,
_rtDW -> TransportDelay_IWORK . CircularBufSize , & _rtDW ->
TransportDelay_IWORK . Last , _rtDW -> TransportDelay_IWORK . Tail , _rtDW ->
TransportDelay_IWORK . Head , _rtP -> P_24 , 0 , ( boolean_T ) (
ssIsMinorTimeStep ( S ) && ( ( * uBuffer + _rtDW -> TransportDelay_IWORK .
CircularBufSize ) [ _rtDW -> TransportDelay_IWORK . Head ] == ssGetT ( S ) )
) ) ; } isHit = ssIsSampleHit ( S , 1 , 0 ) ; if ( isHit != 0 ) { _rtB ->
B_4_3_0 = _rtP -> P_25 ; _rtB -> B_4_4_0 = _rtDW -> Memory_PreviousInput ; }
if ( ssGetT ( S ) >= _rtB -> B_4_3_0 ) { _rtB -> B_4_7_0 = ( _rtB -> B_4_1_0
- B_4_2_0 ) * _rtP -> P_21 ; } else { _rtB -> B_4_7_0 = _rtB -> B_4_4_0 ; }
_rtB -> B_4_8_0 = _rtB -> B_12_109_0 [ 0 ] * _rtB -> B_12_109_0 [ 0 ] ; if (
_rtB -> B_4_7_0 > _rtP -> P_27 ) { rtb_B_12_155_0 = _rtP -> P_27 ; } else if
( _rtB -> B_4_7_0 < _rtP -> P_28 ) { rtb_B_12_155_0 = _rtP -> P_28 ; } else {
rtb_B_12_155_0 = _rtB -> B_4_7_0 ; } _rtB -> B_4_10_0 = muDoubleScalarSqrt (
rtb_B_12_155_0 ) ; if ( ssIsModeUpdateTimeStep ( S ) ) { srUpdateBC ( _rtDW
-> TrueRMS_SubsysRanBC ) ; } } isHit = ssIsSampleHit ( S , 1 , 0 ) ; if ( (
isHit != 0 ) && ssIsModeUpdateTimeStep ( S ) ) { if ( _rtB -> B_12_25_0_m ) {
if ( ! _rtDW -> RMS_MODE ) { if ( ssGetTaskTime ( S , 1 ) != ssGetTStart ( S
) ) { ssSetBlockStateForSolverChangedAtMajorStep ( S ) ; } _rtDW -> RMS_MODE
= true ; } } else if ( _rtDW -> RMS_MODE ) {
ssSetBlockStateForSolverChangedAtMajorStep ( S ) ; _rtDW -> RMS_MODE = false
; } } if ( _rtDW -> RMS_MODE ) { _rtB -> B_2_0_0 = _rtX ->
integrator_CSTATE_l ; { real_T * * uBuffer = ( real_T * * ) & _rtDW ->
TransportDelay_PWORK_i . TUbufferPtrs [ 0 ] ; real_T simTime = ssGetT ( S ) ;
real_T tMinusDelay = simTime - _rtP -> P_3 ; B_2_1_0 =
AC_Feeder_Control_acc_rt_TDelayInterpolate ( tMinusDelay , 0.0 , * uBuffer ,
_rtDW -> TransportDelay_IWORK_j . CircularBufSize , & _rtDW ->
TransportDelay_IWORK_j . Last , _rtDW -> TransportDelay_IWORK_j . Tail ,
_rtDW -> TransportDelay_IWORK_j . Head , _rtP -> P_4 , 0 , ( boolean_T ) (
ssIsMinorTimeStep ( S ) && ( ( * uBuffer + _rtDW -> TransportDelay_IWORK_j .
CircularBufSize ) [ _rtDW -> TransportDelay_IWORK_j . Head ] == ssGetT ( S )
) ) ) ; } isHit = ssIsSampleHit ( S , 1 , 0 ) ; if ( isHit != 0 ) { _rtB ->
B_2_3_0 = _rtP -> P_5 ; } rtb_B_2_4_0 = ( ssGetT ( S ) >= _rtB -> B_2_3_0 ) ;
isHit = ssIsSampleHit ( S , 1 , 0 ) ; if ( isHit != 0 ) { _rtB -> B_2_5_0 =
_rtDW -> Memory_PreviousInput_h ; } if ( rtb_B_2_4_0 ) { _rtB -> B_2_7_0 = (
_rtB -> B_2_0_0 - B_2_1_0 ) * _rtP -> P_1 ; } else { _rtB -> B_2_7_0 = _rtB
-> B_2_5_0 ; } _rtB -> B_2_8_0 = _rtX -> integrator_CSTATE_k ; { real_T * *
uBuffer = ( real_T * * ) & _rtDW -> TransportDelay_PWORK_i3 . TUbufferPtrs [
0 ] ; real_T simTime = ssGetT ( S ) ; real_T tMinusDelay = simTime - _rtP ->
P_8 ; B_2_9_0 = AC_Feeder_Control_acc_rt_TDelayInterpolate ( tMinusDelay ,
0.0 , * uBuffer , _rtDW -> TransportDelay_IWORK_a . CircularBufSize , & _rtDW
-> TransportDelay_IWORK_a . Last , _rtDW -> TransportDelay_IWORK_a . Tail ,
_rtDW -> TransportDelay_IWORK_a . Head , _rtP -> P_9 , 0 , ( boolean_T ) (
ssIsMinorTimeStep ( S ) && ( ( * uBuffer + _rtDW -> TransportDelay_IWORK_a .
CircularBufSize ) [ _rtDW -> TransportDelay_IWORK_a . Head ] == ssGetT ( S )
) ) ) ; } isHit = ssIsSampleHit ( S , 1 , 0 ) ; if ( isHit != 0 ) { _rtB ->
B_2_11_0 = _rtP -> P_10 ; } rtb_B_2_4_0 = ( ssGetT ( S ) >= _rtB -> B_2_11_0
) ; isHit = ssIsSampleHit ( S , 1 , 0 ) ; if ( isHit != 0 ) { _rtB ->
B_2_13_0 = _rtDW -> Memory_PreviousInput_hn ; } if ( rtb_B_2_4_0 ) { _rtB ->
B_2_15_0 = ( _rtB -> B_2_8_0 - B_2_9_0 ) * _rtP -> P_0 ; } else { _rtB ->
B_2_15_0 = _rtB -> B_2_13_0 ; } _rtB -> B_2_19_0 = ( muDoubleScalarSin ( _rtP
-> P_14 * ssGetTaskTime ( S , 0 ) + _rtP -> P_15 ) * _rtP -> P_12 + _rtP ->
P_13 ) * _rtB -> B_12_109_0 [ 0 ] ; _rtB -> B_2_21_0 = ( muDoubleScalarSin (
_rtP -> P_18 * ssGetTaskTime ( S , 0 ) + _rtP -> P_19 ) * _rtP -> P_16 + _rtP
-> P_17 ) * _rtB -> B_12_109_0 [ 0 ] ; _rtB -> B_2_22_0 = _rtP -> P_20 *
muDoubleScalarHypot ( _rtB -> B_2_7_0 , _rtB -> B_2_15_0 ) ; if (
ssIsModeUpdateTimeStep ( S ) ) { srUpdateBC ( _rtDW -> RMS_SubsysRanBC ) ; }
} if ( _rtB -> B_12_24_0 ) { _rtB -> B_12_117_0 = _rtB -> B_4_10_0 ; } else {
_rtB -> B_12_117_0 = _rtB -> B_2_22_0 ; } isHit = ssIsSampleHit ( S , 1 , 0 )
; if ( isHit != 0 ) { if ( ssIsModeUpdateTimeStep ( S ) ) { if ( ( ( _rtZCE
-> Sampling_Trig_ZCE == POS_ZCSIG ) != ( int32_T ) _rtB -> B_12_21_0_g ) && (
_rtZCE -> Sampling_Trig_ZCE != UNINITIALIZED_ZCSIG ) ) { _rtB -> B_5_0_0 =
_rtB -> B_12_117_0 ; if ( ssGetLogOutput ( S ) ) { { const char * errMsg = (
NULL ) ; void * fp = ( void * ) _rtDW -> ToFile_PWORK . FilePtr ; if ( fp !=
( NULL ) ) { { real_T t ; void * u ; t = ssGetTaskTime ( S , 1 ) ; u = ( void
* ) & _rtB -> B_5_0_0 ; errMsg = rtwH5LoggingCollectionWrite ( 1 , fp , 0 , t
, u ) ; if ( errMsg != ( NULL ) ) { ssSetErrorStatus ( S , errMsg ) ; return
; } } } } } _rtDW -> Sampling_SubsysRanBC = 4 ; } _rtZCE -> Sampling_Trig_ZCE
= _rtB -> B_12_21_0_g ; } ssCallAccelRunBlock ( S , 12 , 119 ,
SS_CALL_MDL_OUTPUTS ) ; _rtB -> B_12_154_0 = _rtP -> P_78 ; } if (
ssGetTaskTime ( S , 0 ) < _rtP -> P_79 ) { rtb_B_12_155_0 = _rtP -> P_80 ; }
else { rtb_B_12_155_0 = _rtP -> P_81 ; } isHit = ssIsSampleHit ( S , 1 , 0 )
; if ( isHit != 0 ) { _rtB -> B_12_157_0 = rt_Lookup ( & _rtP -> P_82 [ 0 ] ,
4 , ssGetTaskTime ( S , 1 ) , & _rtP -> P_83 [ 0 ] ) ; } if ( ! ( _rtB ->
B_12_26_0 >= _rtP -> P_84 ) ) { rtb_B_12_155_0 = _rtB -> B_12_157_0 ; } isHit
= ssIsSampleHit ( S , 1 , 0 ) ; if ( isHit != 0 ) { _rtB -> B_12_160_0 =
rt_Lookup ( & _rtP -> P_85 [ 0 ] , 5 , ssGetTaskTime ( S , 1 ) , & _rtP ->
P_86 [ 0 ] ) ; } if ( _rtB -> B_12_154_0 >= _rtP -> P_87 ) { if ( _rtB ->
B_12_27_0_c >= _rtP -> P_29 ) { _rtB -> B_12_162_0 = rtb_B_12_155_0 ; } else
{ _rtB -> B_12_162_0 = _rtB -> B_12_30_0 ; } } else { _rtB -> B_12_162_0 =
_rtB -> B_12_160_0 ; } isHit = ssIsSampleHit ( S , 1 , 0 ) ; if ( isHit != 0
) { _rtB -> B_12_166_0 = _rtP -> P_88 ; _rtB -> B_12_168_0 = rt_Lookup ( &
_rtP -> P_89 [ 0 ] , 5 , ssGetTaskTime ( S , 1 ) , & _rtP -> P_90 [ 0 ] ) ; }
if ( _rtB -> B_12_166_0 >= _rtP -> P_91 ) { if ( _rtB -> B_12_28_0 >= _rtP ->
P_30 ) { _rtB -> B_12_170_0 = rtb_B_12_155_0 ; } else { _rtB -> B_12_170_0 =
_rtB -> B_12_30_0 ; } } else { _rtB -> B_12_170_0 = _rtB -> B_12_168_0 ; }
isHit = ssIsSampleHit ( S , 1 , 0 ) ; if ( isHit != 0 ) { _rtB -> B_12_174_0
= _rtP -> P_92 ; _rtB -> B_12_176_0 = rt_Lookup ( & _rtP -> P_93 [ 0 ] , 5 ,
ssGetTaskTime ( S , 1 ) , & _rtP -> P_94 [ 0 ] ) ; } if ( _rtB -> B_12_174_0
>= _rtP -> P_95 ) { if ( _rtB -> B_12_29_0_b >= _rtP -> P_31 ) { _rtB ->
B_12_178_0 = rtb_B_12_155_0 ; } else { _rtB -> B_12_178_0 = _rtB -> B_12_30_0
; } } else { _rtB -> B_12_178_0 = _rtB -> B_12_176_0 ; } isHit =
ssIsSampleHit ( S , 1 , 0 ) ; if ( isHit != 0 ) { _rtB -> B_12_189_0 = _rtP
-> P_96 ; } if ( ssGetTaskTime ( S , 0 ) < _rtP -> P_97 ) { rtb_B_12_155_0 =
_rtP -> P_98 ; } else { rtb_B_12_155_0 = _rtP -> P_99 ; } isHit =
ssIsSampleHit ( S , 1 , 0 ) ; if ( isHit != 0 ) { _rtB -> B_12_192_0 =
rt_Lookup ( & _rtP -> P_100 [ 0 ] , 4 , ssGetTaskTime ( S , 1 ) , & _rtP ->
P_101 [ 0 ] ) ; } if ( ! ( _rtB -> B_12_31_0_p >= _rtP -> P_102 ) ) {
rtb_B_12_155_0 = _rtB -> B_12_192_0 ; } isHit = ssIsSampleHit ( S , 1 , 0 ) ;
if ( isHit != 0 ) { _rtB -> B_12_195_0 = rt_Lookup ( & _rtP -> P_103 [ 0 ] ,
5 , ssGetTaskTime ( S , 1 ) , & _rtP -> P_104 [ 0 ] ) ; } if ( _rtB ->
B_12_189_0 >= _rtP -> P_105 ) { if ( _rtB -> B_12_32_0 >= _rtP -> P_32 ) {
_rtB -> B_12_197_0 = rtb_B_12_155_0 ; } else { _rtB -> B_12_197_0 = _rtB ->
B_12_35_0 ; } } else { _rtB -> B_12_197_0 = _rtB -> B_12_195_0 ; } isHit =
ssIsSampleHit ( S , 1 , 0 ) ; if ( isHit != 0 ) { _rtB -> B_12_201_0 = _rtP
-> P_106 ; _rtB -> B_12_203_0 = rt_Lookup ( & _rtP -> P_107 [ 0 ] , 5 ,
ssGetTaskTime ( S , 1 ) , & _rtP -> P_108 [ 0 ] ) ; } if ( _rtB -> B_12_201_0
>= _rtP -> P_109 ) { if ( _rtB -> B_12_33_0_c >= _rtP -> P_33 ) { _rtB ->
B_12_205_0 = rtb_B_12_155_0 ; } else { _rtB -> B_12_205_0 = _rtB -> B_12_35_0
; } } else { _rtB -> B_12_205_0 = _rtB -> B_12_203_0 ; } isHit =
ssIsSampleHit ( S , 1 , 0 ) ; if ( isHit != 0 ) { _rtB -> B_12_209_0 = _rtP
-> P_110 ; _rtB -> B_12_211_0 = rt_Lookup ( & _rtP -> P_111 [ 0 ] , 5 ,
ssGetTaskTime ( S , 1 ) , & _rtP -> P_112 [ 0 ] ) ; } if ( _rtB -> B_12_209_0
>= _rtP -> P_113 ) { if ( _rtB -> B_12_34_0 >= _rtP -> P_34 ) { _rtB ->
B_12_213_0 = rtb_B_12_155_0 ; } else { _rtB -> B_12_213_0 = _rtB -> B_12_35_0
; } } else { _rtB -> B_12_213_0 = _rtB -> B_12_211_0 ; } UNUSED_PARAMETER (
tid ) ; } static void mdlOutputsTID2 ( SimStruct * S , int_T tid ) {
B_AC_Feeder_Control_T * _rtB ; DW_AC_Feeder_Control_T * _rtDW ;
P_AC_Feeder_Control_T * _rtP ; int32_T i ; _rtDW = ( ( DW_AC_Feeder_Control_T
* ) ssGetRootDWork ( S ) ) ; _rtP = ( ( P_AC_Feeder_Control_T * )
ssGetModelRtp ( S ) ) ; _rtB = ( ( B_AC_Feeder_Control_T * )
_ssGetModelBlockIO ( S ) ) ; _rtB -> B_12_0_0 = _rtP -> P_114 ; _rtB ->
B_12_4_0 = _rtP -> P_117 ; _rtB -> B_12_7_0 = _rtP -> P_120 ; _rtB ->
B_12_8_0 = _rtP -> P_121 ; _rtB -> B_12_5_0 [ 0 ] = _rtP -> P_118 [ 0 ] ;
_rtB -> B_12_9_0 [ 0 ] = _rtP -> P_122 * _rtP -> P_119 [ 0 ] ; _rtB ->
B_12_5_0 [ 1 ] = _rtP -> P_118 [ 1 ] ; _rtB -> B_12_9_0 [ 1 ] = _rtP -> P_122
* _rtP -> P_119 [ 1 ] ; _rtB -> B_12_5_0 [ 2 ] = _rtP -> P_118 [ 2 ] ; _rtB
-> B_12_9_0 [ 2 ] = _rtP -> P_122 * _rtP -> P_119 [ 2 ] ; _rtB -> B_12_10_0 =
( _rtP -> P_115 * _rtB -> B_12_0_0 + _rtP -> P_116 ) * _rtP -> P_123 ; _rtB
-> B_12_11_0_m = _rtP -> P_124 ; _rtB -> B_12_12_0 = _rtP -> P_125 ; _rtB ->
B_12_13_0 = _rtP -> P_126 ; _rtB -> B_12_14_0 = _rtP -> P_127 ; _rtB ->
B_12_15_0_c = _rtP -> P_128 ; _rtB -> B_12_16_0 = _rtP -> P_129 ; _rtB ->
B_12_17_0 = _rtP -> P_144 ; _rtB -> B_12_18_0 = _rtP -> P_145 ; _rtB ->
B_12_19_0 = _rtP -> P_130 ; _rtB -> B_12_20_0_k = _rtP -> P_131 ; _rtB ->
B_12_21_0_g = _rtP -> P_146 ; { if ( _rtDW ->
TAQSigLogging_InsertedFor_Clk_at_outport_0_PWORK . AQHandles &&
ssGetLogOutput ( S ) ) { sdiWriteSignal ( _rtDW ->
TAQSigLogging_InsertedFor_Clk_at_outport_0_PWORK . AQHandles , ssGetTaskTime
( S , 2 ) , ( char * ) & _rtB -> B_12_21_0_g + 0 ) ; } } _rtB -> B_12_24_0 =
( _rtP -> P_132 != 0.0 ) ; _rtB -> B_12_25_0_m = ! _rtB -> B_12_24_0 ; _rtB
-> B_12_26_0 = _rtP -> P_133 ; _rtB -> B_12_27_0_c = _rtP -> P_134 ; _rtB ->
B_12_28_0 = _rtP -> P_135 ; _rtB -> B_12_29_0_b = _rtP -> P_136 ; _rtB ->
B_12_30_0 = _rtP -> P_137 ; _rtB -> B_12_31_0_p = _rtP -> P_138 ; _rtB ->
B_12_32_0 = _rtP -> P_139 ; _rtB -> B_12_33_0_c = _rtP -> P_140 ; _rtB ->
B_12_34_0 = _rtP -> P_141 ; _rtB -> B_12_35_0 = _rtP -> P_142 ; for ( i = 0 ;
i < 6 ; i ++ ) { _rtB -> B_12_36_0_f [ i ] = _rtP -> P_143 [ i ] ; }
UNUSED_PARAMETER ( tid ) ; }
#define MDL_UPDATE
static void mdlUpdate ( SimStruct * S , int_T tid ) { B_AC_Feeder_Control_T *
_rtB ; DW_AC_Feeder_Control_T * _rtDW ; P_AC_Feeder_Control_T * _rtP ;
int32_T isHit ; _rtDW = ( ( DW_AC_Feeder_Control_T * ) ssGetRootDWork ( S ) )
; _rtP = ( ( P_AC_Feeder_Control_T * ) ssGetModelRtp ( S ) ) ; _rtB = ( (
B_AC_Feeder_Control_T * ) _ssGetModelBlockIO ( S ) ) ; isHit = ssIsSampleHit
( S , 1 , 0 ) ; if ( isHit != 0 ) { _rtDW -> UnitDelay_DSTATE = _rtB ->
B_12_11_0 ; _rtDW -> DiscreteTimeIntegrator1_DSTATE += _rtP -> P_36 * _rtB ->
B_12_7_0 ; _rtDW -> DiscreteTimeIntegrator1_PrevResetState = ( int8_T ) _rtB
-> B_12_10_0_g ; _rtDW -> UnitDelay_DSTATE_e = _rtB -> B_12_22_0 ; _rtDW ->
DiscreteTimeIntegrator1_DSTATE_l += _rtP -> P_38 * _rtB -> B_12_71_0 ; _rtDW
-> DiscreteTimeIntegrator1_PrevResetState_m = ( int8_T ) _rtB -> B_12_21_0 ;
ssCallAccelRunBlock ( S , 12 , 46 , SS_CALL_MDL_UPDATE ) ; } if ( _rtDW ->
TrueRMS_MODE ) { { real_T * * uBuffer = ( real_T * * ) & _rtDW ->
TransportDelay_PWORK . TUbufferPtrs [ 0 ] ; real_T simTime = ssGetT ( S ) ;
_rtDW -> TransportDelay_IWORK . Head = ( ( _rtDW -> TransportDelay_IWORK .
Head < ( _rtDW -> TransportDelay_IWORK . CircularBufSize - 1 ) ) ? ( _rtDW ->
TransportDelay_IWORK . Head + 1 ) : 0 ) ; if ( _rtDW -> TransportDelay_IWORK
. Head == _rtDW -> TransportDelay_IWORK . Tail ) { if ( !
AC_Feeder_Control_acc_rt_TDelayUpdateTailOrGrowBuf ( & _rtDW ->
TransportDelay_IWORK . CircularBufSize , & _rtDW -> TransportDelay_IWORK .
Tail , & _rtDW -> TransportDelay_IWORK . Head , & _rtDW ->
TransportDelay_IWORK . Last , simTime - _rtP -> P_23 , uBuffer , ( boolean_T
) 0 , false , & _rtDW -> TransportDelay_IWORK . MaxNewBufSize ) ) {
ssSetErrorStatus ( S , "tdelay memory allocation error" ) ; return ; } } ( *
uBuffer + _rtDW -> TransportDelay_IWORK . CircularBufSize ) [ _rtDW ->
TransportDelay_IWORK . Head ] = simTime ; ( * uBuffer ) [ _rtDW ->
TransportDelay_IWORK . Head ] = _rtB -> B_4_1_0 ; } isHit = ssIsSampleHit ( S
, 1 , 0 ) ; if ( isHit != 0 ) { _rtDW -> Memory_PreviousInput = _rtB ->
B_4_7_0 ; } } if ( _rtDW -> RMS_MODE ) { { real_T * * uBuffer = ( real_T * *
) & _rtDW -> TransportDelay_PWORK_i . TUbufferPtrs [ 0 ] ; real_T simTime =
ssGetT ( S ) ; _rtDW -> TransportDelay_IWORK_j . Head = ( ( _rtDW ->
TransportDelay_IWORK_j . Head < ( _rtDW -> TransportDelay_IWORK_j .
CircularBufSize - 1 ) ) ? ( _rtDW -> TransportDelay_IWORK_j . Head + 1 ) : 0
) ; if ( _rtDW -> TransportDelay_IWORK_j . Head == _rtDW ->
TransportDelay_IWORK_j . Tail ) { if ( !
AC_Feeder_Control_acc_rt_TDelayUpdateTailOrGrowBuf ( & _rtDW ->
TransportDelay_IWORK_j . CircularBufSize , & _rtDW -> TransportDelay_IWORK_j
. Tail , & _rtDW -> TransportDelay_IWORK_j . Head , & _rtDW ->
TransportDelay_IWORK_j . Last , simTime - _rtP -> P_3 , uBuffer , ( boolean_T
) 0 , false , & _rtDW -> TransportDelay_IWORK_j . MaxNewBufSize ) ) {
ssSetErrorStatus ( S , "tdelay memory allocation error" ) ; return ; } } ( *
uBuffer + _rtDW -> TransportDelay_IWORK_j . CircularBufSize ) [ _rtDW ->
TransportDelay_IWORK_j . Head ] = simTime ; ( * uBuffer ) [ _rtDW ->
TransportDelay_IWORK_j . Head ] = _rtB -> B_2_0_0 ; } isHit = ssIsSampleHit (
S , 1 , 0 ) ; if ( isHit != 0 ) { _rtDW -> Memory_PreviousInput_h = _rtB ->
B_2_7_0 ; } { real_T * * uBuffer = ( real_T * * ) & _rtDW ->
TransportDelay_PWORK_i3 . TUbufferPtrs [ 0 ] ; real_T simTime = ssGetT ( S )
; _rtDW -> TransportDelay_IWORK_a . Head = ( ( _rtDW ->
TransportDelay_IWORK_a . Head < ( _rtDW -> TransportDelay_IWORK_a .
CircularBufSize - 1 ) ) ? ( _rtDW -> TransportDelay_IWORK_a . Head + 1 ) : 0
) ; if ( _rtDW -> TransportDelay_IWORK_a . Head == _rtDW ->
TransportDelay_IWORK_a . Tail ) { if ( !
AC_Feeder_Control_acc_rt_TDelayUpdateTailOrGrowBuf ( & _rtDW ->
TransportDelay_IWORK_a . CircularBufSize , & _rtDW -> TransportDelay_IWORK_a
. Tail , & _rtDW -> TransportDelay_IWORK_a . Head , & _rtDW ->
TransportDelay_IWORK_a . Last , simTime - _rtP -> P_8 , uBuffer , ( boolean_T
) 0 , false , & _rtDW -> TransportDelay_IWORK_a . MaxNewBufSize ) ) {
ssSetErrorStatus ( S , "tdelay memory allocation error" ) ; return ; } } ( *
uBuffer + _rtDW -> TransportDelay_IWORK_a . CircularBufSize ) [ _rtDW ->
TransportDelay_IWORK_a . Head ] = simTime ; ( * uBuffer ) [ _rtDW ->
TransportDelay_IWORK_a . Head ] = _rtB -> B_2_8_0 ; } isHit = ssIsSampleHit (
S , 1 , 0 ) ; if ( isHit != 0 ) { _rtDW -> Memory_PreviousInput_hn = _rtB ->
B_2_15_0 ; } } UNUSED_PARAMETER ( tid ) ; }
#define MDL_UPDATE
static void mdlUpdateTID2 ( SimStruct * S , int_T tid ) { UNUSED_PARAMETER (
tid ) ; }
#define MDL_DERIVATIVES
static void mdlDerivatives ( SimStruct * S ) { B_AC_Feeder_Control_T * _rtB ;
DW_AC_Feeder_Control_T * _rtDW ; XDot_AC_Feeder_Control_T * _rtXdot ; _rtDW =
( ( DW_AC_Feeder_Control_T * ) ssGetRootDWork ( S ) ) ; _rtXdot = ( (
XDot_AC_Feeder_Control_T * ) ssGetdX ( S ) ) ; _rtB = ( (
B_AC_Feeder_Control_T * ) _ssGetModelBlockIO ( S ) ) ; _rtXdot ->
Integrator1_CSTATE = _rtB -> B_12_67_0 ; _rtXdot -> Integrator_CSTATE = _rtB
-> B_12_79_0 ; if ( _rtDW -> TrueRMS_MODE ) { _rtXdot -> integrator_CSTATE =
_rtB -> B_4_8_0 ; } else { ( ( XDot_AC_Feeder_Control_T * ) ssGetdX ( S ) )
-> integrator_CSTATE = 0.0 ; } if ( _rtDW -> RMS_MODE ) { _rtXdot ->
integrator_CSTATE_l = _rtB -> B_2_19_0 ; _rtXdot -> integrator_CSTATE_k =
_rtB -> B_2_21_0 ; } else { { real_T * dx ; int_T i ; dx = & ( ( (
XDot_AC_Feeder_Control_T * ) ssGetdX ( S ) ) -> integrator_CSTATE_l ) ; for (
i = 0 ; i < 2 ; i ++ ) { dx [ i ] = 0.0 ; } } } } static void
mdlInitializeSizes ( SimStruct * S ) { ssSetChecksumVal ( S , 0 , 4033798546U
) ; ssSetChecksumVal ( S , 1 , 3677963111U ) ; ssSetChecksumVal ( S , 2 ,
1129923667U ) ; ssSetChecksumVal ( S , 3 , 1533071062U ) ; { mxArray *
slVerStructMat = ( NULL ) ; mxArray * slStrMat = mxCreateString ( "simulink"
) ; char slVerChar [ 10 ] ; int status = mexCallMATLAB ( 1 , & slVerStructMat
, 1 , & slStrMat , "ver" ) ; if ( status == 0 ) { mxArray * slVerMat =
mxGetField ( slVerStructMat , 0 , "Version" ) ; if ( slVerMat == ( NULL ) ) {
status = 1 ; } else { status = mxGetString ( slVerMat , slVerChar , 10 ) ; }
} mxDestroyArray ( slStrMat ) ; mxDestroyArray ( slVerStructMat ) ; if ( (
status == 1 ) || ( strcmp ( slVerChar , "10.7" ) != 0 ) ) { return ; } }
ssSetOptions ( S , SS_OPTION_EXCEPTION_FREE_CODE ) ; if ( ssGetSizeofDWork (
S ) != ( SLSize ) sizeof ( DW_AC_Feeder_Control_T ) ) { static char msg [ 256
] ; sprintf ( msg , "Unexpected error: Internal DWork sizes do "
"not match for accelerator mex file (%ld vs %lu)." , ( signed long )
ssGetSizeofDWork ( S ) , ( unsigned long ) sizeof ( DW_AC_Feeder_Control_T )
) ; ssSetErrorStatus ( S , msg ) ; } if ( ssGetSizeofGlobalBlockIO ( S ) != (
SLSize ) sizeof ( B_AC_Feeder_Control_T ) ) { static char msg [ 256 ] ;
sprintf ( msg , "Unexpected error: Internal BlockIO sizes do "
"not match for accelerator mex file (%ld vs %lu)." , ( signed long )
ssGetSizeofGlobalBlockIO ( S ) , ( unsigned long ) sizeof (
B_AC_Feeder_Control_T ) ) ; ssSetErrorStatus ( S , msg ) ; } { int
ssSizeofParams ; ssGetSizeofParams ( S , & ssSizeofParams ) ; if (
ssSizeofParams != sizeof ( P_AC_Feeder_Control_T ) ) { static char msg [ 256
] ; sprintf ( msg , "Unexpected error: Internal Parameters sizes do "
"not match for accelerator mex file (%d vs %lu)." , ssSizeofParams , (
unsigned long ) sizeof ( P_AC_Feeder_Control_T ) ) ; ssSetErrorStatus ( S ,
msg ) ; } } _ssSetModelRtp ( S , ( real_T * ) & AC_Feeder_Control_rtDefaultP
) ; rt_InitInfAndNaN ( sizeof ( real_T ) ) ; ( ( P_AC_Feeder_Control_T * )
ssGetModelRtp ( S ) ) -> P_27 = rtInf ; } static void
mdlInitializeSampleTimes ( SimStruct * S ) { slAccRegPrmChangeFcn ( S ,
mdlOutputsTID2 ) ; } static void mdlTerminate ( SimStruct * S ) { }
#include "simulink.c"
