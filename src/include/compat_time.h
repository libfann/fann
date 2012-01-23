/*

Originally timeval.h by Wu Yongwei

Fast Artificial Neural Network Library (fann)
Copyright (C) 2003-2012 Steffen Nissen (sn@leenissen.dk)

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/  
	
/*
 * timeval.h    1.0 01/12/19
 *
 * Defines gettimeofday, timeval, etc. for Win32
 *
 * By Wu Yongwei
 *
 */ 
	
#ifndef _TIMEVAL_H
#define _TIMEVAL_H
	
#ifdef _WIN32
	
/* Modified to compile as ANSI C without include of windows.h
   If this gives problems with future Windows/MSC versions, then
   uncomment the USE_WINDOWS_H definition to switch back. */ 
/* #define USE_WINDOWS_H */ 
#ifdef USE_WINDOWS_H
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else	/*  */
#ifndef _INC_WINDOWS
#define VOID void
#define WINAPI __stdcall
#define OUT
#define WINBASEAPI
typedef long LONG;
typedef unsigned long DWORD;
typedef __int64 LONGLONG;
typedef struct _FILETIME
{
	DWORD dwLowDateTime;
	DWORD dwHighDateTime;
} FILETIME, *LPFILETIME;
typedef union _LARGE_INTEGER
{
	
		/* Removed unnamed struct,
		 * it is not ANSI C compatible */ 
		/* struct {
		 * DWORD LowPart;
		 * LONG HighPart;
		 * }; */ 
		struct
	{
		DWORD LowPart;
		LONG HighPart;
	} u;
	 LONGLONG QuadPart;
 } LARGE_INTEGER;

WINBASEAPI VOID WINAPI  GetSystemTimeAsFileTime(OUT LPFILETIME lpSystemTimeAsFileTime);

#endif	/* _INC_WINDOWS */
#endif	/* USE_WINDOWS_H */
	
#include <time.h>
	
#ifndef __GNUC__
#define EPOCHFILETIME (116444736000000000i64)
#else	/*  */
#define EPOCHFILETIME (116444736000000000LL)
#endif	/*  */
	struct timeval
{
	long tv_sec;				/* seconds */
	  long tv_usec;			/* microseconds */
 };
struct timezone
{
	int tz_minuteswest;		/* minutes W of Greenwich */
	  int tz_dsttime;			/* type of dst correction */
 };
__inline int gettimeofday(struct timeval *tv, struct timezone *tz) 
{
	FILETIME ft;
	LARGE_INTEGER li;
	__int64 t;
	static int tzflag;

	if(tv)
		
	{
		GetSystemTimeAsFileTime(&ft);
		
			/* The following two lines have been modified to use the named
			 * union member. Unnamed members are not ANSI C compatible. */ 
			li.u.LowPart = ft.dwLowDateTime;
		li.u.HighPart = ft.dwHighDateTime;
		t = li.QuadPart;		/* In 100-nanosecond intervals */
		t -= EPOCHFILETIME;	/* Offset to the Epoch time */
		t /= 10;				/* In microseconds */
		tv->tv_sec = (long) (t / 1000000);
		tv->tv_usec = (long) (t % 1000000);
	}
	if(tz)
		
	{
		if(!tzflag)
			
		{
			_tzset();
			tzflag++;
		}
		tz->tz_minuteswest = _timezone / 60;
		tz->tz_dsttime = _daylight;
	}
	return 0;
}


#else	/* _WIN32 */
	
#include <sys/time.h>
	
#endif	/* _WIN32 */
	
#endif	/* _TIMEVAL_H */
