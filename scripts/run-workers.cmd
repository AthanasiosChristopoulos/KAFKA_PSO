@echo off
setlocal

cd /d "%~dp0..\java" || exit /b 1

@REM mvn -q -e -DskipTests -Dexec.mainClass=pso.WorkerGroupMain -Pcpu compile exec:java
@REM mvn -q -e -DskipTests -Dexec.mainClass=pso.WorkerGroupMain ^
@REM -Dorg.slf4j.simpleLogger.defaultLogLevel=debug ^
@REM -Dorg.slf4j.simpleLogger.log.org.deeplearning4j=debug ^
@REM -Dorg.slf4j.simpleLogger.log.org.nd4j=debug ^
@REM -Pcpu compile exec:java

mvn -e -DskipTests -Dexec.mainClass=pso.WorkerGroupMain ^
-Dorg.slf4j.simpleLogger.defaultLogLevel=error ^
-Pcpu compile exec:java
endlocal
