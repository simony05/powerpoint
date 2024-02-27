import React from 'react'
import { useSession, signIn, signOut } from 'next-auth/react'
 
export default function Navbar() {
    const { data, status } = useSession();
    if (status === 'loading') return (
        <nav>
            <div>
                Loading...
            </div>
        </nav>
    )
    if (status === 'authenticated') {
        return (
            <nav>
                <div>
                    <button class="google" onClick={signOut}>Sign out</button>
                </div>
            </nav>
        );
    }
    return (
        <nav>
            <div>
                <button class="google" onClick={() => signIn('google')}>Sign in with Google</button>
            </div>
        </nav>
    );
}